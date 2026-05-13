import os
from ultralytics import YOLO

def export_to_onnx(model_path):
    print(f"Loading weights from {model_path}...")
    
    if not os.path.exists(model_path):
        print("Error: Weight file not found. Ensure training completed successfully.")
        return
        
    model = YOLO(model_path)
    
    print("Exporting model to ONNX format...")
    # Export the model to ONNX format, optimizing for dynamic batching on edge devices
    success = model.export(format="onnx", dynamic=True, simplify=True)
    
    print(f"Export successful: {success}")

if __name__ == "__main__":
    # Point to the best weights from our recent training run
    best_weights_path = "C:\\Users\\PC GAMING\\breaker-detection-project\\runs\\detect\\model\\runs\\train-3\\weights\\best.pt"
    export_to_onnx(best_weights_path)
