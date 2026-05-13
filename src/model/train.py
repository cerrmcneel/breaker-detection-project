# src/model/train.py
from ultralytics import YOLO


def init_model():
    """Load YOLO26-Nano (NMS-free) as the base model for transfer learning."""
    model = YOLO("yolo26n.pt")
    return model


def run_training():
    model = init_model()

    print("Starting YOLO26-Nano Training Pipeline...")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=1280,
        batch=4,
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
        
        # YOLO26 Specific Architecture Features
        # Note: STAL and ProgLoss are internal architectural features of YOLO26
        # and are enabled by default within the model graph. They are not
        # configurable training hyperparameters.
    )


if __name__ == "__main__":
    run_training()
