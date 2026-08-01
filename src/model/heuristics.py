def compute_iou(box1, box2):
    """Calculates Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

class SpatialHeuristicEngine:
    def __init__(self, mainbreaker_position="leftmost"):
        """
        Initializes the heuristic engine.
        Args:
            mainbreaker_position (str): "leftmost", "rightmost", "topmost", or "bottommost"
        """
        self.mainbreaker_position = mainbreaker_position
        from src.model.hmm_corrector import HMMCorrector
        self.hmm_corrector = HMMCorrector()

    def group_into_rows(self, predictions, y_tolerance=50):
        """Groups bounding boxes into physical DIN-rail rows based on Y coordinates."""
        if not predictions:
            return []
            
        # Sort by Y-coordinate (top to bottom)
        sorted_by_y = sorted(predictions, key=lambda p: p['box'][1])
        
        rows = []
        current_row = [sorted_by_y[0]]
        
        for pred in sorted_by_y[1:]:
            # If the Y difference is within tolerance, it's on the same rail
            if abs(pred['box'][1] - current_row[-1]['box'][1]) <= y_tolerance:
                current_row.append(pred)
            else:
                rows.append(current_row)
                current_row = [pred]
        rows.append(current_row)
        
        # Sort each row left-to-right
        for row in rows:
            row.sort(key=lambda p: p['box'][0])
            
        return rows

    def apply_logic(self, predictions, image_path=None, use_hmm=True):
        """
        Ingests raw YOLO predictions and reclassifies based on HMM sequence decoding.
        Expected format: [{'box': [x1, y1, x2, y2], 'class': 'MCB', 'conf': 0.88}, ...]
        """
        if not predictions:
            return []

        rows = self.group_into_rows(predictions)

        refined_rows = []
        for rail_idx, row in enumerate(rows):
            rail_type = "main" if rail_idx == 0 else "secondary"
            
            # 1. Find baseline module width in pixels BEFORE merging or extrapolating
            # general household MCBs are the smallest, 1-module components and form the majority
            widths = []
            for p in row:
                w_pixels = p['box'][2] - p['box'][0]
                if p['class'] in ['RCD', 'RCD_SI', 'MAINBREAKER', 'OVERSURGE']:
                    widths.append(w_pixels / 2.0)
                else:
                    widths.append(w_pixels)
            if widths:
                widths.sort()
                # Use 25th percentile to find the 1-module size, protecting against large misclassified units
                median_width = widths[max(0, len(widths) // 4)]
                if median_width < 15:
                    median_width = 40.0
            else:
                median_width = 40.0

            # 2. Fragmented Box Merging (merge overlapping boxes of the same class)
            merged_row = []
            if row:
                curr = row[0]
                for next_p in row[1:]:
                    # Use 2D IoU to check for significant overlap (e.g. double detections of the same object)
                    iou = compute_iou(curr['box'], next_p['box'])
                    if iou > 0.4 and next_p['class'] == curr['class']:
                        # Merge boxes
                        curr['box'] = [
                            min(curr['box'][0], next_p['box'][0]),
                            min(curr['box'][1], next_p['box'][1]),
                            max(curr['box'][2], next_p['box'][2]),
                            max(curr['box'][3], next_p['box'][3])
                        ]
                        curr['conf'] = max(curr['conf'], next_p['conf'])
                        curr['heuristic_applied'] = True
                        curr['heuristic_correction'] = 'MERGE_FRAGMENTED'
                    else:
                        merged_row.append(curr)
                        curr = next_p
                merged_row.append(curr)
            row = merged_row
                
            # 3. Grid Extrapolation (The "Missing Box" Fixer)
            i = 0
            while i < len(row) - 1:
                curr = row[i]
                next_p = row[i+1]
                gap = next_p['box'][0] - curr['box'][2]
                
                # If gap is roughly large enough to fit an MCB
                if gap > (median_width * 1.1):
                    synthetic_box = [
                        curr['box'][2] + 2, 
                        curr['box'][1],     
                        curr['box'][2] + 2 + median_width, 
                        curr['box'][3]      
                    ]
                    # Don't let it overlap the next device
                    if synthetic_box[2] > next_p['box'][0]:
                        synthetic_box[2] = next_p['box'][0] - 2
                        
                    synthetic_pred = {
                        'box': synthetic_box,
                        'class': 'MCB',
                        'conf': 0.50, # Indicate it was heuristically generated
                        'heuristic_applied': True,
                        'heuristic_correction': 'GRID_EXTRAPOLATION'
                    }
                    row.insert(i+1, synthetic_pred)
                i += 1

            if use_hmm:
                # 4. Construct HMM observation sequence
                hmm_sequence = []
                for pred in row:
                    w_pixels = pred["box"][2] - pred["box"][0]
                    mod_width = max(1, min(4, round(w_pixels / median_width)))
                    
                    hmm_sequence.append({
                        "class": pred["class"],
                        "conf": pred["conf"],
                        "width": mod_width,
                        "ocr_text": pred.get("ocr_text", "")
                    })

                # 5. Decode sequence using HMM Viterbi corrector
                corrected_states = self.hmm_corrector.decode_rail(hmm_sequence, rail_type)
                
                # 6. Apply corrected states back to predictions
                for idx, state in enumerate(corrected_states):
                    orig_class = row[idx]["class"]
                    if orig_class != state:
                        row[idx]["class"] = state
                        row[idx]["heuristic_applied"] = True
                        row[idx]["heuristic_correction"] = "HMM_VITERBI_DECODE"
                        row[idx]["hmm_original_class"] = orig_class
                    
            refined_rows.extend(row)

        return refined_rows

# Example usage for testing
if __name__ == "__main__":
    engine = SpatialHeuristicEngine()
    
    # Mock output on main rail representing:
    # 1. Main Breaker (detected as MCB with high confidence - size 2 modules)
    # 2. RCD (clean)
    # 3. MCB (clean)
    # 4. MCB (clean)
    mock_yolo_output = [
        {'box': [100, 50, 180, 200], 'class': 'MCB', 'conf': 0.95}, # Should become MAINBREAKER due to size 2 and transition
        {'box': [200, 50, 280, 200], 'class': 'RCD', 'conf': 0.88}, 
        {'box': [290, 50, 330, 200], 'class': 'MCB', 'conf': 0.91},
        {'box': [340, 50, 380, 200], 'class': 'MCB', 'conf': 0.92}
    ]
    
    refined = engine.apply_logic(mock_yolo_output)
    print("Refined Predictions:")
    for p in refined:
        changed_marker = f" (Corrected from {p['hmm_original_class']})" if p.get("heuristic_correction") == "HMM_VITERBI_DECODE" else ""
        print(f"  Class: {p['class']:12s} | Box: {p['box']}{changed_marker}")
