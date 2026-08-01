# src/model/train_hmm.py
import collections
import json
import os

from src.data_gen.grammar import PanelFactory


def train_transition_matrix(num_simulations=10000, output_path="src/model/hmm_config.json"):
    """
    Simulates thousands of compliant and chaotic panels to compute the transition
    and initial state probabilities for our HMM.
    """
    print(f"Running {num_simulations} panel simulations to train HMM...")
    
    # Initialize the panel factory with chaos to learn both compliant and non-compliant transitions
    factory = PanelFactory(chaos_factor=0.5, boost_minority=True)
    
    # Count variables
    # transition_counts[state_i][state_j] = count
    transition_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    # initial_counts[rail_type][state] = count  (rail_type: 'main' vs 'secondary')
    initial_counts = {"main": collections.defaultdict(int), "secondary": collections.defaultdict(int)}
    
    for _ in range(num_simulations):
        panel = factory.generate()
        for rail_idx, rail in enumerate(panel.rails):
            rail_type = "main" if rail_idx == 0 else "secondary"
            components = [comp.cls for comp in rail.components]
            
            if not components:
                continue
                
            # Count initial state (first element on the rail)
            first_state = components[0]
            initial_counts[rail_type][first_state] += 1
            
            # Count transitions (State A -> State B)
            for i in range(len(components) - 1):
                state_i = components[i]
                state_j = components[i+1]
                transition_counts[state_i][state_j] += 1
                
    # --- TASK: MATHEMATICAL NORMALIZATION & SMOOTHING ---
    # We must normalize the counts into probabilities.
    # To prevent zero-probability errors (which break Viterbi when encountering
    # unusual layouts), we should apply Laplace Smoothing (add a tiny value, e.g., alpha=0.01).
    
    classes = ["MCB", "RCD", "RCD_SI", "MAINBREAKER", "OVERSURGE", "OTHER"]
    alpha = 0.01 # Laplace smoothing parameter
    
    transition_matrix = {}
    initial_probs = {"main": {}, "secondary": {}}
    
    # 1. Normalize Transitions
    for state_i in classes:
        transition_matrix[state_i] = {}
        total_i = sum(transition_counts[state_i][state_j] for state_j in classes)
        
        for state_j in classes:
            count = transition_counts[state_i].get(state_j, 0)
            # Apply Laplace smoothing
            prob = (count + alpha) / (total_i + len(classes) * alpha)
            transition_matrix[state_i][state_j] = prob
        
    # 2. Normalize Initial State Probabilities
    for rail_type in ["main", "secondary"]:
        total_initial = sum(initial_counts[rail_type][state] for state in classes)
        
        for state in classes:
            count = initial_counts[rail_type].get(state, 0)
            # Apply Laplace smoothing
            prob = (count + alpha) / (total_initial + len(classes) * alpha)
            initial_probs[rail_type][state] = prob

    # Save to config JSON
    config = {
        "transition_matrix": transition_matrix,
        "initial_probs": initial_probs
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"HMM configuration successfully trained and saved to {output_path}")

    # Auto-update the HMM concept document in our OKF bundle (Option A)
    import datetime

    from src.tools.okf_updater import update_frontmatter
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    okf_hmm_path = os.path.join(project_root, "docs", "okf", "models", "hmm_decoder.md")
    
    if os.path.exists(okf_hmm_path):
        current_time = datetime.datetime.now().astimezone().isoformat()
        updates = {
            "timestamp": current_time,
            "metrics": {
                "laplace_smoothing_alpha": alpha,
                "states_count": len(classes),
                "simulations_trained": num_simulations
            }
        }
        try:
            update_frontmatter(okf_hmm_path, updates)
            print(f"Auto-updated OKF concept documentation at: {okf_hmm_path}")
        except Exception as e:
            print(f"Warning: Failed to auto-update OKF file: {e}")

if __name__ == "__main__":
    train_transition_matrix()
