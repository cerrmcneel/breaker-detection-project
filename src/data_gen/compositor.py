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

            # Paste onto canvas with edge-feathered blending
            bg_slice = canvas[
                ann["y"]: ann["y"] + ann["h"],
                ann["x"]: ann["x"] + ann["w"],
            ]
            canvas[
                ann["y"]: ann["y"] + ann["h"],
                ann["x"]: ann["x"] + ann["w"],
            ] = self._blend_seed_feathered(bg_slice, seed_resized)

        # Apply shadows and specular glare to the composite canvas to simulate basement conditions
        if augment:
            import random
            # Apply linear shadows with 50% probability
            if random.random() < 0.5:
                canvas = self._apply_shadows(canvas)
            # Apply specular glare with 50% probability
            if random.random() < 0.5:
                canvas = self._apply_glare(canvas)

        # Apply perspective warp with a 70% probability to increase training diversity
        if augment:
            import random
            if random.random() < 0.7:
                canvas, annotations = self._apply_perspective_warp(canvas, annotations)

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

    @staticmethod
    def _blend_seed_feathered(bg_slice, seed_resized, feather_px=4):
        """
        Blends seed_resized into bg_slice using a soft feathered mask around the edges
        to avoid sharp, artificial copy-paste boundaries.
        """
        h, w = seed_resized.shape[:2]
        import numpy as np
        if h <= feather_px * 2 or w <= feather_px * 2:
            return seed_resized # Too small to feather
            
        # Create a 2D mask of 1s (representing seed content)
        mask = np.ones((h, w), dtype=np.float32)
        
        # Feather horizontal/vertical edges
        for i in range(feather_px):
            alpha = (i + 1) / (feather_px + 1)
            mask[i, :] *= alpha          # Top edge
            mask[h - 1 - i, :] *= alpha  # Bottom edge
            mask[:, i] *= alpha          # Left edge
            mask[:, w - 1 - i] *= alpha  # Right edge
            
        # Blur the mask slightly to make transitions even smoother
        import cv2
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Add channel dimension
        mask_3d = np.expand_dims(mask, axis=-1)
        
        # Perform linear interpolation blending
        blended = (seed_resized.astype(np.float32) * mask_3d) + (bg_slice.astype(np.float32) * (1.0 - mask_3d))
        return blended.astype(np.uint8)

    @staticmethod
    def _apply_perspective_warp(canvas, annotations, max_warp=80):
        """
        Applies a random 3D perspective homography to the entire canvas,
        and re-maps all bounding box annotations to fit the new perspective.
        """
        import cv2
        import numpy as np
        import random
        
        h, w = canvas.shape[:2]
        
        # Define the original corners of the canvas
        src_pts = np.array([
            [0, 0],         # Top-Left
            [w - 1, 0],     # Top-Right
            [w - 1, h - 1], # Bottom-Right
            [0, h - 1]      # Bottom-Left
        ], dtype=np.float32)
        
        # Apply random perspective shifts to define target corners
        top_left_shift_x = random.randint(-max_warp, max_warp)
        top_left_shift_y = random.randint(-max_warp, max_warp)
        top_right_shift_x = random.randint(-max_warp, max_warp)
        top_right_shift_y = random.randint(-max_warp, max_warp)
        bottom_right_shift_x = random.randint(-max_warp, max_warp)
        bottom_right_shift_y = random.randint(-max_warp, max_warp)
        bottom_left_shift_x = random.randint(-max_warp, max_warp)
        bottom_left_shift_y = random.randint(-max_warp, max_warp)
        
        dst_pts = np.array([
            [top_left_shift_x, top_left_shift_y],
            [w - 1 + top_right_shift_x, top_right_shift_y],
            [w - 1 + bottom_right_shift_x, h - 1 + bottom_right_shift_y],
            [bottom_left_shift_x, h - 1 + bottom_left_shift_y]
        ], dtype=np.float32)
        
        # Get homography matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Warp the canvas image with a realistic dark gray background
        warped_canvas = cv2.warpPerspective(canvas, M, (w, h), borderValue=(30, 30, 30))
        
        # Warp all annotations
        warped_annotations = []
        for ann in annotations:
            # Reconstruct the 4 corners of the bounding box
            bx1 = ann["x"]
            by1 = ann["y"]
            bx2 = ann["x"] + ann["w"]
            by2 = ann["y"] + ann["h"]
            
            box_pts = np.array([
                [bx1, by1],
                [bx2, by1],
                [bx2, by2],
                [bx1, by2]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            # Project box corners
            warped_box_pts = cv2.perspectiveTransform(box_pts, M).reshape(-1, 2)
            
            # Find the new bounding box enclosing these 4 projected points
            min_x = np.min(warped_box_pts[:, 0])
            max_x = np.max(warped_box_pts[:, 0])
            min_y = np.min(warped_box_pts[:, 1])
            max_y = np.max(warped_box_pts[:, 1])
            
            # Clamp to canvas dimensions
            min_x = max(0, min(w - 1, min_x))
            max_x = max(0, min(w - 1, max_x))
            min_y = max(0, min(h - 1, min_y))
            max_y = max(0, min(h - 1, max_y))
            
            new_w = max_x - min_x
            new_h = max_y - min_y
            
            # Keep only valid annotations
            if new_w > 5 and new_h > 5:
                warped_annotations.append({
                    "class_id": ann["class_id"],
                    "x": int(min_x),
                    "y": int(min_y),
                    "w": int(new_w),
                    "h": int(new_h)
                })
                
        return warped_canvas, warped_annotations

    @staticmethod
    def _apply_shadows(img):
        """
        Superimposes a random linear shadow gradient (e.g., top cutout shadow)
        to simulate uneven overhead lighting.
        """
        import numpy as np
        import random
        
        h, w = img.shape[:2]
        direction = random.choice(["top", "bottom", "left", "right"])
        
        shadow_ratio = random.uniform(0.3, 0.7)
        max_opacity = random.uniform(0.4, 0.75)
        
        if direction in ["top", "bottom"]:
            length = int(h * shadow_ratio)
            gradient = np.linspace(max_opacity, 0.0, length, dtype=np.float32)
            mask = np.ones((h,), dtype=np.float32)
            if direction == "top":
                mask[:length] = 1.0 - gradient
            else:
                mask[h - length:] = 1.0 - np.flip(gradient)
            mask_3d = mask[:, np.newaxis, np.newaxis]
        else:
            length = int(w * shadow_ratio)
            gradient = np.linspace(max_opacity, 0.0, length, dtype=np.float32)
            mask = np.ones((w,), dtype=np.float32)
            if direction == "left":
                mask[:length] = 1.0 - gradient
            else:
                mask[w - length:] = 1.0 - np.flip(gradient)
            mask_3d = mask[np.newaxis, :, np.newaxis]
            
        img_float = img.astype(np.float32) * mask_3d
        return np.clip(img_float, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_glare(img):
        """
        Superimposes a radial glare highlight at a random location
        to simulate camera flash or direct light reflections on glossy plastic.
        """
        import numpy as np
        import random
        
        h, w = img.shape[:2]
        
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        
        diag = np.sqrt(w**2 + h**2)
        r = int(diag * random.uniform(0.15, 0.40))
        
        y, x = np.ogrid[:h, :w]
        dist_sq = (x - cx)**2 + (y - cy)**2
        
        sigma = r / 2.0
        mask = np.exp(-dist_sq / (2 * (sigma**2)))
        
        max_glare = random.uniform(60, 180)
        glare_overlay = (mask * max_glare).astype(np.float32)
        
        b_mult = random.uniform(0.85, 1.0)
        g_mult = random.uniform(0.90, 1.0)
        r_mult = random.uniform(0.95, 1.0)
        
        glare_3d = np.zeros_like(img, dtype=np.float32)
        glare_3d[..., 0] = glare_overlay * b_mult
        glare_3d[..., 1] = glare_overlay * g_mult
        glare_3d[..., 2] = glare_overlay * r_mult
        
        img_float = img.astype(np.float32) + glare_3d
        return np.clip(img_float, 0, 255).astype(np.uint8)
