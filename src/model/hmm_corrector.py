# src/model/hmm_corrector.py
import os
import json
import numpy as np

class HMMCorrector:
    def __init__(self, config_path="src/model/hmm_config.json"):
        # 1. Load the trained HMM parameters (transitions and priors)
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                self.transition_matrix = config["transition_matrix"]
                self.initial_probs = config["initial_probs"]
        else:
            print(f"Warning: {config_path} not found. Run train_hmm.py first.")
            self.transition_matrix = {}
            self.initial_probs = {"main": {}, "secondary": {}}

        self.classes = ["MCB", "RCD", "RCD_SI", "MAINBREAKER", "OVERSURGE", "OTHER"]

        # 2. Width Prior: P(width | state)
        #
        # IMPORTANT — two realities in tension here:
        #
        #   Physical reality (REBT domain knowledge):
        #     MCBs are predominantly 2-module in modern Spanish panels.
        #     RCDs/RCD_SI are always 2-module.
        #     Comb OVERSURGE units are typically 3-4 modules.
        #     Single-phase IGA (MAINBREAKER) is 2 modules (being phased out).
        #
        #   Computed reality (heuristics.py output):
        #     The pixel→module conversion anchors on the 25th percentile of
        #     all box widths on a rail. If most boxes ARE 2-module MCBs, the
        #     anchor equals the MCB pixel width, so every MCB computes to
        #     mod_width=1 and every RCD (wider) computes to mod_width=2.
        #
        # The values below are calibrated to the COMPUTED distribution, not
        # the physical one. Any change here must be accompanied by a fix to
        # the heuristic's anchor estimation in heuristics.py.
        self.width_priors = {
            "MCB":         {1: 0.80, 2: 0.20},                    # heuristic outputs 1 for most MCBs
            "RCD":         {2: 1.00},                              # always computes to 2 (wider than MCBs)
            "RCD_SI":      {2: 1.00},                              # same form factor as RCD
            "MAINBREAKER": {2: 0.70, 3: 0.15, 4: 0.15},           # single-phase IGA still most common
            "OVERSURGE":   {2: 0.50, 3: 0.30, 4: 0.20},           # physical: 3-4 mod; heuristic: unclear
            "OTHER":       {1: 0.40, 2: 0.40, 3: 0.10, 4: 0.10}
        }
        # Hard penalty for widths not in the dict. DO NOT raise without also
        # fixing the heuristic — softening this (e.g. to 0.05) removes MCB/RCD
        # discriminating power and drops classification by ~3pp.
        self._width_fallback = 0.001

        # 3. YOLO Confusion Matrix: P(yolo_class | state)
        # Represents model error rates (visual similarity confusion).
        # These defaults are overridden at runtime if calibrate_hmm_confusion.py
        # has written an empirical `yolo_confusion` key into hmm_config.json.
        self.yolo_confusion = {
            "MCB":         {"MCB": 0.85, "RCD": 0.05, "RCD_SI": 0.03, "MAINBREAKER": 0.02, "OVERSURGE": 0.01, "OTHER": 0.04},
            "RCD":         {"MCB": 0.05, "RCD": 0.75, "RCD_SI": 0.15, "MAINBREAKER": 0.02, "OVERSURGE": 0.01, "OTHER": 0.02},
            "RCD_SI":      {"MCB": 0.05, "RCD": 0.15, "RCD_SI": 0.75, "MAINBREAKER": 0.02, "OVERSURGE": 0.01, "OTHER": 0.02},
            "MAINBREAKER": {"MCB": 0.02, "RCD": 0.02, "RCD_SI": 0.01, "MAINBREAKER": 0.80, "OVERSURGE": 0.10, "OTHER": 0.05},
            "OVERSURGE":   {"MCB": 0.02, "RCD": 0.02, "RCD_SI": 0.01, "MAINBREAKER": 0.10, "OVERSURGE": 0.80, "OTHER": 0.05},
            "OTHER":       {"MCB": 0.05, "RCD": 0.02, "RCD_SI": 0.01, "MAINBREAKER": 0.02, "OVERSURGE": 0.02, "OTHER": 0.88}
        }

        # Override with empirically calibrated confusion if available
        if "yolo_confusion" in config:
            self.yolo_confusion = config["yolo_confusion"]
            print("HMMCorrector: loaded empirical YOLO confusion matrix from config.")
        else:
            print("HMMCorrector: using default hand-coded YOLO confusion matrix.")

    def _get_emission_probability(self, state, yolo_class, yolo_conf, width, ocr_text):
        """
        Calculates P(observation | state) using our multi-modal products:
        P(YOLO | state) * P(width | state) * P(OCR | state)
        """
        # A. Visual probability (YOLO confusion & confidence)
        p_visual = self.yolo_confusion.get(state, {}).get(yolo_class, 0.01)
        
        # B. Spatial probability (Module Width constraints)
        width_dist = self.width_priors.get(state, {})
        p_width = width_dist.get(width, self._width_fallback)

        # C. Textual probability (OCR feedback)
        p_ocr = 1.0
        if ocr_text:
            cleaned_ocr = ocr_text.upper().strip()
            
            # Case 1: SI indicator (Superinmunizado RCD)
            if cleaned_ocr == "SI":
                p_ocr = 0.95 if state == "RCD_SI" else 0.05
            # Case 2: Standard MCB Curve (e.g. C16, B20)
            elif any(cleaned_ocr.startswith(c) for c in ["B", "C", "D"]) and any(char.isdigit() for char in cleaned_ocr):
                p_ocr = 0.95 if state == "MCB" else 0.05
            # Case 3: Leakage current (e.g. 30MA)
            elif "30MA" in cleaned_ocr or "0.03A" in cleaned_ocr:
                p_ocr = 0.95 if state in ["RCD", "RCD_SI"] else 0.05
                
        # Return total multi-modal emission probability
        return p_visual * yolo_conf * p_width * p_ocr

    def decode_rail(self, sequence, rail_type="secondary"):
        """
        Performs bidirectional Viterbi decoding. Decodes the sequence left-to-right
        and right-to-left, returning the state path with the highest probability.
        
        sequence: list of dicts: [
            {"class": "MCB", "conf": 0.92, "width": 1, "ocr_text": "C16", "box": [...]}, ...
        ]
        """
        if not sequence:
            return []

        # 1. Decode Left-to-Right
        states_l2r, prob_l2r = self._viterbi_decode(sequence, rail_type)
        
        # 2. Decode Right-to-Left (reverse sequence, decode, then reverse back)
        reversed_sequence = list(reversed(sequence))
        states_r2l_rev, prob_r2l = self._viterbi_decode(reversed_sequence, rail_type)
        states_r2l = list(reversed(states_r2l_rev))

        # 3. Choose the sequence with the higher likelihood
        if prob_l2r >= prob_r2l:
            return states_l2r
        else:
            return states_r2l

    def _viterbi_decode(self, sequence, rail_type):
        """
        Custom, dependency-free Viterbi algorithm.
        Returns:
            best_path (list of str): decoded state sequence
            path_probability (float): total path likelihood
        """
        T = len(sequence)
        N = len(self.classes)
        
        # Initialize DP tables
        # V[t][j] = float probability
        V = [{c: 0.0 for c in self.classes} for _ in range(T)]
        # ptr[t][j] = str name of previous state
        ptr = [{c: None for c in self.classes} for _ in range(T)]
        
        # Define safe log helper to avoid log(0) warnings
        def safe_log(p):
            return np.log(p) if p > 1e-12 else -100.0

        # 1. INITIALIZATION (t = 0)
        first_obs = sequence[0]
        for state in self.classes:
            prior = self.initial_probs.get(rail_type, {}).get(state, 0.01)
            emission = self._get_emission_probability(
                state, first_obs["class"], first_obs["conf"], first_obs["width"], first_obs.get("ocr_text")
            )
            V[0][state] = safe_log(prior) + safe_log(emission)
            
        # 2. RECURSION (t = 1 to T-1)
        for t in range(1, T):
            obs = sequence[t]
            for j in self.classes:
                emission = self._get_emission_probability(
                    j, obs["class"], obs["conf"], obs["width"], obs.get("ocr_text")
                )
                log_emission = safe_log(emission)
                
                max_val = -float('inf')
                best_prev = None
                for i in self.classes:
                    trans = self.transition_matrix.get(i, {}).get(j, 0.01)
                    val = V[t-1][i] + safe_log(trans)
                    if val > max_val:
                        max_val = val
                        best_prev = i
                        
                V[t][j] = max_val + log_emission
                ptr[t][j] = best_prev
        
        # 3. TERMINATION
        max_final_val = -float('inf')
        best_final_state = None
        for state in self.classes:
            val = V[T-1][state]
            if val > max_final_val:
                max_final_val = val
                best_final_state = state
                
        path_probability = np.exp(max_final_val) if max_final_val > -100.0 else 0.0
        
        # 4. BACKTRACKING
        best_path = [None] * T
        best_path[T-1] = best_final_state
        
        for t in range(T-2, -1, -1):
            next_state = best_path[t+1]
            best_path[t] = ptr[t+1][next_state]
            
        return best_path, path_probability

if __name__ == "__main__":
    # Test stub
    corrector = HMMCorrector()
    print("HMM Corrector loaded.\n")
    
    # Mock sequence representing a typical rail:
    # 1. Main Breaker (clean)
    # 2. RCD (clean)
    # 3. An MCB misclassified as RCD (conf 0.70, but width 1 and OCR "C16" - typical visual error)
    # 4. MCB (clean)
    # 5. MCB (clean)
    mock_sequence = [
        {"class": "MAINBREAKER", "conf": 0.90, "width": 2, "ocr_text": ""},
        {"class": "RCD", "conf": 0.85, "width": 2, "ocr_text": ""},
        {"class": "RCD", "conf": 0.70, "width": 1, "ocr_text": "C16"},
        {"class": "MCB", "conf": 0.95, "width": 1, "ocr_text": ""},
        {"class": "MCB", "conf": 0.95, "width": 1, "ocr_text": ""}
    ]
    
    print("Input Sequence:")
    for idx, item in enumerate(mock_sequence):
        print(f"  {idx+1}. Class: {item['class']:12s} | Conf: {item['conf']:.2f} | Width: {item['width']} | OCR: '{item['ocr_text']}'")
        
    corrected = corrector.decode_rail(mock_sequence, rail_type="main")
    
    print("\nCorrected Sequence (HMM output):")
    for idx, cls in enumerate(corrected):
        orig = mock_sequence[idx]['class']
        changed_marker = " *CHANGED*" if cls != orig else ""
        print(f"  {idx+1}. Class: {cls:12s}{changed_marker}")
