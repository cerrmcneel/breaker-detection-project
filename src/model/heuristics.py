class SpatialHeuristicEngine:
    def __init__(self, mainbreaker_position="leftmost"):
        """
        Initializes the heuristic engine.
        Args:
            mainbreaker_position (str): "leftmost", "rightmost", "topmost", or "bottommost"
        """
        self.mainbreaker_position = mainbreaker_position

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

    def apply_logic(self, predictions, image_path=None):
        """
        Ingests raw YOLO predictions and reclassifies based on electrical layout rules.
        Expected format: [{'box': [x1, y1, x2, y2], 'class': 'MCB', 'conf': 0.88}, ...]
        """
        if not predictions:
            return []

        rows = self.group_into_rows(predictions)
        all_sorted = [pred for row in rows for pred in row]
        
        import statistics
        
        # Add Grid Extrapolation (The "Missing Box" Fixer)
        for row in rows:
            mcb_widths = [p['box'][2] - p['box'][0] for p in row if p['class'] == 'MCB']
            if not mcb_widths:
                continue
            median_width = statistics.median(mcb_widths)
            
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

        # Re-flatten the list after adding synthetic boxes
        all_sorted = [pred for row in rows for pred in row]

        # Rule 8: The Sandwiched RCD Failsafe
        # RCDs are almost never placed in the middle of a continuous block of MCBs.
        # If an RCD has an MCB immediately to its left AND an MCB immediately to its right,
        # it is almost certainly a YOLO false positive.
        for row in rows:
            for i in range(1, len(row) - 1):
                curr = row[i]
                if curr['class'] in ['RCD', 'RCD_SI']:
                    left_dev = row[i-1]
                    right_dev = row[i+1]
                    if left_dev['class'] == 'MCB' and right_dev['class'] == 'MCB':
                        # Check horizontal proximity to ensure they are actually adjacent
                        dist_left = curr['box'][0] - left_dev['box'][2]
                        dist_right = right_dev['box'][0] - curr['box'][2]
                        if dist_left < 30 and dist_right < 30:
                            curr['class'] = 'MCB'
                            curr['heuristic_applied'] = True
                            curr['heuristic_correction'] = 'SANDWICHED_RCD_DEMOTION'

        # Rule 5: RCD Cluster Sanity Check (The "5 RCDs" fix)
        # If we see multiple RCDs in a row without MCBs, the extras are likely MCBs.
        # This is a common YOLO error due to the visual similarity of test buttons.
        rcd_count = 0
        for i, pred in enumerate(all_sorted):
            if pred['class'] in ['RCD', 'RCD_SI']:
                rcd_count += 1
                # If we've already seen an RCD and this one is right next to it, 
                # be suspicious if it's very narrow.
                if rcd_count > 1:
                    # Heuristic: Only the first RCD in a sequence is likely a real RCD
                    # The others are likely MCBs misidentified.
                    pred['class'] = 'MCB'
                    pred['conf'] = pred['conf'] * 0.8 # Lower confidence of the correction
                    pred['heuristic_correction'] = "RCD_CLUSTER_FIX"
            else:
                rcd_count = 0 # Reset count when we hit an MCB or other

        # Rule 6: Fragmented Box Merging (The "Oversurge" fix)
        # If two boxes of the same class are almost touching, merge them.
        final_preds = []
        if all_sorted:
            curr = all_sorted[0]
            for next_p in all_sorted[1:]:
                # Check if boxes are almost touching (within 15 pixels)
                dist = next_p['box'][0] - curr['box'][2]
                if dist < 15 and next_p['class'] == curr['class']:
                    # Merge boxes
                    curr['box'] = [
                        min(curr['box'][0], next_p['box'][0]),
                        min(curr['box'][1], next_p['box'][1]),
                        max(curr['box'][2], next_p['box'][2]),
                        max(curr['box'][3], next_p['box'][3])
                    ]
                    curr['conf'] = max(curr['conf'], next_p['conf'])
                    curr['heuristic_applied'] = True
                else:
                    final_preds.append(curr)
                    curr = next_p
            final_preds.append(curr)

        # Rule 7: Global RCD Mathematical Cap
        # The REBT recommends no more than 5 MCBs per RCD. We enforce a global mathematical cap.
        # This will demote the lowest confidence RCDs if the model over-predicts them.
        import math
        total_mcbs = sum(1 for p in final_preds if p['class'] == 'MCB')
        max_rcds = max(1, math.ceil(total_mcbs / 5.0))
        
        all_rcds = [p for p in final_preds if p['class'] in ['RCD', 'RCD_SI']]
        if len(all_rcds) > max_rcds:
            # Sort by confidence ascending (lowest confidence first)
            all_rcds.sort(key=lambda x: x['conf'])
            
            # Demote the excess RCDs to MCBs
            excess_count = len(all_rcds) - max_rcds
            for i in range(excess_count):
                rcd_to_demote = all_rcds[i]
                rcd_to_demote['class'] = 'MCB'
                rcd_to_demote['heuristic_applied'] = True
                rcd_to_demote['heuristic_correction'] = 'GLOBAL_RCD_CAP'

        # Rule 9: Test Button Scanner (OpenCV Classic Vision)
        # Verify RCDs by scanning for the physical test button using Canny Edge Detection.
        if image_path:
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                for pred in final_preds:
                    if pred['class'] in ['RCD', 'RCD_SI']:
                        x1, y1, x2, y2 = map(int, pred['box'])
                        # Clamp coordinates to image boundaries
                        h, w = img.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        crop = img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                            
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        edges = cv2.Canny(blurred, 50, 150)
                        
                        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        
                        button_found = False
                        total_area = (x2 - x1) * (y2 - y1)
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            # Test button is usually 0.5% to 15% of the breaker's surface area
                            if total_area * 0.005 < area < total_area * 0.15:
                                button_found = True
                                break
                                
                        if not button_found:
                            pred['class'] = 'MCB'
                            pred['heuristic_applied'] = True
                            pred['heuristic_correction'] = 'NO_BUTTON_DETECTED'

        return final_preds

# Example usage for testing
if __name__ == "__main__":
    engine = SpatialHeuristicEngine()
    
    # Mock output: 
    # Row 1: MCB (IGA), Gap, RCD, MCB, MCB
    mock_yolo_output = [
        {'box': [100, 50, 150, 200], 'class': 'MCB', 'conf': 0.95}, # Should become MAINBREAKER
        {'box': [250, 50, 300, 200], 'class': 'RCD', 'conf': 0.88}, # Anchor
        {'box': [310, 50, 360, 200], 'class': 'MCB', 'conf': 0.91}, # Normal MCB
        {'box': [370, 50, 420, 200], 'class': 'MCB', 'conf': 0.92}  # Normal MCB
    ]
    
    refined = engine.apply_logic(mock_yolo_output)
    print("Refined Predictions:")
    for p in refined:
        print(f"{p['class']} at x={p['box'][0]}")
