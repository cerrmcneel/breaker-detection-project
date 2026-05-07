class Compositor:
    """
    Converts a Grammar Panel into a synthetic training image + annotations.

    Design
    ------
    Two separate responsibilities are kept intentionally isolated:

    1. calculate_layout(panel) — PURE function.
       Walks the Panel → Rails → Breakers tree and converts
       module-width integers into pixel bounding boxes.
       No images, no cv2, fully unit-testable.

    2. compose(panel) — IMPURE function.
       Uses calculate_layout, then pastes seed images onto a blank
       canvas via cv2/numpy. Requires cv2 to be installed.

    Parameters
    ----------
    seed_library : SeedLibrary | None
        Provides random cropped images for each breaker class.
        Can be None when only calculate_layout is needed (e.g. tests).
    img_width : int
        Canvas width in pixels (default: 640 for YOLO-standard).
    img_height : int
        Canvas height in pixels (default: 640 for YOLO-standard).
    module_width_px : int
        Pixel width of one electrical module slot on the panel.
        A MainBreaker (width=4) will be 4 * module_width_px wide.
    """

    # Maps breaker class names → integer YOLO class IDs.
    # Final Spanish panel taxonomy (confirmed 2026-05-04):
    #   MCB          — PIA (Pequeño Interruptor Automático), 1–2 modules
    #   RCD          — Diferencial AC-type (legacy), 30mA, 2 modules
    #   RCD_SI       — Diferencial Superinmunizado (Type A/F), DC-sensitive, 30mA
    #   MAINBREAKER  — IGA (Interruptor General Automático), 1–4 modules
    #   OVERSURGE    — IGA+DPS (integrated surge protection), 2–4 modules
    #   OTHER        — Timers, contactors, 300mA fire-protection RCDs, unclassified
    CLASS_MAP = {
        "MCB":         0,
        "RCD":         1,
        "RCD_SI":      2,
        "MAINBREAKER": 3,
        "OVERSURGE":   4,
        "OTHER":       5,
    }

    def __init__(self, seed_library, img_width=640, img_height=640, module_width_px=40):
        self.seed_library    = seed_library
        self.img_width       = img_width
        self.img_height      = img_height
        self.module_width_px = module_width_px

    # ------------------------------------------------------------------
    # 1. PURE — pixel coordinate layout calculator
    # ------------------------------------------------------------------
    def calculate_layout(self, panel):
        """
        Walk the Panel and return a flat list of annotation dicts.

        Each dict contains pixel-space bounding box coordinates:
            {class_id, x, y, w, h}

        These are raw pixels — call normalize_box (label_writer.py)
        before writing to a YOLO .txt file.

        Parameters
        ----------
        panel : Panel
            A Grammar Panel with rails_count rails, each holding
            Breaker components.

        Returns
        -------
        list[dict]
            One dict per breaker, in rail-order then left-to-right order.
        """
        annotations = []
        
        # Calculate vertical offset to center the entire panel
        total_height = len(panel.rails) * panel.rail_height
        start_y = max(0, (self.img_height - int(total_height)) // 2)

        for rail_idx, rail in enumerate(panel.rails):
            y = start_y + int(rail_idx * panel.rail_height)
            h = int(panel.rail_height)

            # Calculate total width of this rail to center it horizontally
            total_rail_width = sum([b.width * self.module_width_px for b in rail.components])
            cursor_x = max(0, (self.img_width - int(total_rail_width)) // 2)

            for breaker in rail.components:
                w = int(breaker.width * self.module_width_px)
                annotations.append({
                    "class_id": self.CLASS_MAP[breaker.cls],
                    "x": cursor_x,
                    "y": y,
                    "w": w,
                    "h": h,
                })
                cursor_x += w  # advance cursor by this breaker's pixel width

        return annotations

    # ------------------------------------------------------------------
    # 2. IMPURE — image compositor (requires cv2 + a loaded SeedLibrary)
    # ------------------------------------------------------------------
    def compose(self, panel):
        """
        Generate a synthetic panel image by pasting seed crops.

        Parameters
        ----------
        panel : Panel
            Grammar object that defines the layout.

        Returns
        -------
        canvas : np.ndarray
            BGR image of shape (img_height, img_width, 3).
        annotations : list[dict]
            Same format as calculate_layout — pixel bounding boxes.
            Pass this to write_label_file() after normalization.
        """
        import cv2
        import numpy as np

        # Reverse lookup: class_id → class name string
        id_to_cls = {v: k for k, v in self.CLASS_MAP.items()}

        annotations = self.calculate_layout(panel)

        # Black canvas — real panels are dark enclosures
        bg_img = self.seed_library.get_random_background() if hasattr(self.seed_library, 'get_random_background') else None
        
        if bg_img is not None:
            canvas = cv2.resize(bg_img, (self.img_width, self.img_height))
        else:
            canvas = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        for ann in annotations:
            cls_name = id_to_cls[ann["class_id"]]
            seed = self.seed_library.get_random_seed(cls_name)

            # Resize seed crop to the exact bounding box size
            seed_resized = cv2.resize(seed, (ann["w"], ann["h"]))

            # Paste onto canvas — numpy slice is [y:y+h, x:x+w]
            canvas[
                ann["y"]: ann["y"] + ann["h"],
                ann["x"]: ann["x"] + ann["w"],
            ] = seed_resized

        return canvas, annotations
