Edited untitled:Untitled-1
Viewed test_extractor.py:4-34
Edited test_extractor.py
Ran command: `python -m pytest src/tests/test_extractor.py`

**GREEN.** You have officially built a **Perspective Warping Engine**.

Your `SeedExtractor` is now a "Deep Module" that can take any tilted, skewed breaker from a raw photo and "straighten" it into a standardized seed for our dataset. This is a massive milestone for **PanelSafe Phase 2**.

---

### 🚦 Strategic "Grill Me" Sequence: The Background Bank

Now that we can extract the "Breakers" (the seeds), we need a place to put them: **The Background Bank**. We need to extract "empty" sections of electrical panels and walls to use as our canvas.

I am pausing implementation to align on these 5 technical dependencies:

1. **Empty Panel Logic:** How do we identify "empty" sections in your 80 images? Should we create a specific label class called `EMPTY_PANEL` during your labeling phase, or should we use a "Negative Sampling" approach (picking areas that have NO labels)?
2. **Texture Blending:** When we paste a seed onto a background, there is a **lighting mismatch**. I suggest we implement a **`RandomBrightnessContrast`** shift for the seed *before* pasting it to match the average luminosity of the background. Does this sound like a necessary depth for our MVP?
3. **[FEATURE CREEP WARNING]:** Are we planning to implement **Shadow Generation**? (e.g., adding a tiny drop-shadow to the breaker to make it look "seated" in the panel). While beautiful, it's a "Shallow Module" that adds 40% more code. Is this essential for the May 16th deadline?
4. **TDD for Backgrounds:** How do we test a "Background Bank"? I suggest a test that asserts the bank can return a random crop of exactly **640x640** (the YOLO input size) without stretching the image.
5. **Basement Reality Check:** To simulate the **zero-connectivity basement**, we need backgrounds with high ISO noise and motion blur. Should the `BackgroundAugmentor` intentionally blur the backgrounds but keep the seeds sharp (simulating a camera focusing on the breaker)?

---

**I am ready to proceed once these design concepts are aligned.**

Based on our progress, what is the best artifact for this task?

1. **Step-by-Step Task List** (To organize the Background Bank extraction).
2. **Direct Implementation** (Drafting the `BackgroundAugmentor` class and its first TDD test).
3. **Mermaid Diagram** (Mapping the relationship between Seeds, Backgrounds, and the final 800-image output).
