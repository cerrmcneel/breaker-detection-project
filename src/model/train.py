# src/model/train.py
from ultralytics import YOLO


def init_model():
    """Load YOLO26-Large (NMS-free) as the base model for transfer learning.
    Nano -> Medium (2026-07-04, train-18) raised real-world localization recall
    79.34% -> 84.18% at negligible latency cost (101.5ms -> 108.1ms), while
    classification accuracy stayed flat. Trying Large to see if the localization
    trend continues -- localization misses are worse than misclassifications for
    this app, since HITL review can only catch/correct a component that was at
    least detected, not one that was missed entirely."""
    model = YOLO("yolo26l.pt")
    return model


def run_training():
    model = init_model()

    print("Starting YOLO26-Large Training Pipeline...")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=1280,
        batch=2,           # Dropped from 4: Large at batch=4 maxed all 12GB VRAM and
                           # fell into Windows' system-memory-fallback path (0% GPU
                           # utilization at 97% memory used -- PCIe-bound, not compute-bound).
                           # batch=2 should stay fully resident in VRAM and run faster overall.
        device=0,
        workers=0,
        project="model/runs",
        # YOLO26-Nano best practices
        optimizer="auto",    # Uses MuSGD when available
        patience=20,         # Early stopping — prevents overfitting
        cls=1.0,             # Classification loss weight (helps with imbalance)
        
        # --- Sim-to-Real Gap Mitigations ---
        # Augmentations to defeat 2D copy-paste artifacts
        copy_paste=0.5,      # High copy-paste prob to force model to ignore artificial edges
        mosaic=1.0,          # Always use mosaic for context diversity
        mixup=0.2,           # Slight mixup to force texture learning over edge learning
                             # TESTED 2026-07-04 (train-17): disabling mixup + tightening
                             # scale to 0.2 was hypothesized to preserve fine "SI" print
                             # legibility for RCD/RCD_SI. Result: regression, not improvement
                             # (real classification acc 61.73% -> 56.38%, MCB->RCD confusion
                             # 23 -> 26 misclassifications). Reverted. Most likely explanation:
                             # real uploaded photos vary widely in camera distance/framing, so
                             # the scale/mixup augmentation was buying real-world scale/texture
                             # robustness that outweighed the fine-detail-legibility benefit.
        
        # YOLO26 Specific Architecture Features
        # Note: STAL and ProgLoss are internal architectural features of YOLO26
        # and are enabled by default within the model graph. They are not
        # configurable training hyperparameters.
    )


if __name__ == "__main__":
    run_training()
