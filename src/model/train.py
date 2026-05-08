# src/model/train.py
from ultralytics import YOLO

def init_model():
    model = YOLO("yolov8n.pt") 
    return model

def run_training():
    model = init_model()
    
    print("Starting YOLO26 Training Pipeline...")
    
    model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8, 
    device=0, 
    workers=0,
    project="model/runs"
    )
    pass 

if __name__ == "__main__":
    run_training()
