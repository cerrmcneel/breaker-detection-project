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
    def compose(self, panel, augment=True):
        """
        Generate a synthetic panel image by pasting seed crops.

        Parameters
        ----------
        panel : Panel
            Grammar object that defines the layout.
        augment : bool
            If True, apply random per-seed augmentations (brightness,
            rotation, noise) to increase training diversity.

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
            
            # Reverse engineer the module width (e.g. 1, 2, 4) from the pixel width
            module_width = max(1, round(ann["w"] / self.module_width_px))
            
            # Ask the library for a seed of exactly this width (e.g. MCB_2)
            seed = self.seed_library.get_random_seed(cls_name, width=module_width)

            # Aspect-ratio-preserving resize with letterbox padding
            seed_resized = self._aspect_resize_and_pad(seed, ann["w"], ann["h"])

            # Optional per-seed augmentation for training diversity
            if augment:
                seed_resized = self._augment_seed(seed_resized)

            # Paste onto canvas — numpy slice is [y:y+h, x:x+w]
            canvas[
                ann["y"]: ann["y"] + ann["h"],
                ann["x"]: ann["x"] + ann["w"],
            ] = seed_resized

        return canvas, annotations

    # ------------------------------------------------------------------
    # 3. Helpers — resize and augmentation
    # ------------------------------------------------------------------
    @staticmethod
    def _aspect_resize_and_pad(seed, target_w, target_h, pad_color=(30, 30, 30)):
        """
        Resize *seed* to fit inside (target_w, target_h) while preserving
        its original aspect ratio, then center it on a padded background.

        Parameters
        ----------
        seed : np.ndarray
            Source image (BGR, HWC).
        target_w : int
            Target width in pixels.
        target_h : int
            Target height in pixels.
        pad_color : tuple
            BGR color for the letterbox padding.  Dark gray (30,30,30)
            simulates a DIN-rail enclosure background.

        Returns
        -------
        np.ndarray
            Image of exactly (target_h, target_w, 3).
        """
        import cv2
        import numpy as np

        src_h, src_w = seed.shape[:2]

        # Scale factor: fit within target while preserving AR
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))

        # Choose interpolation method based on scaling direction
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        resized = cv2.resize(seed, (new_w, new_h), interpolation=interp)

        # Create padded canvas and center the resized seed
        canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    @staticmethod
    def _augment_seed(img):
        """
        Apply random augmentations to a single seed image.

        Each augmentation fires independently with 50% probability:
          - Brightness shift  ±15%
          - Slight rotation   ±3°
          - Gaussian noise     σ=5

        Parameters
        ----------
        img : np.ndarray
            BGR image to augment (modified in-place where possible).

        Returns
        -------
        np.ndarray
            Augmented image, same shape as input.
        """
        import cv2
        import numpy as np
        import random

        h, w = img.shape[:2]

        # Brightness shift (±15%)
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-0.15, 0.15)
            img = cv2.convertScaleAbs(img, alpha=factor, beta=0)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Forces gray-on-gray test buttons on RCDs to "pop" for the neural network
        if random.random() < 0.5:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Slight rotation (±3°) — simulates camera tilt
        if random.random() < 0.5:
            angle = random.uniform(-3.0, 3.0)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 borderMode=cv2.BORDER_REPLICATE)

        # Gaussian noise (σ=5) — simulates phone camera sensor noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img
