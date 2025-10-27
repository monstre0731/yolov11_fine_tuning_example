'''
conda activate yolov11

'''
from ultralytics import YOLO
import torch # torch will be installed when installing ultralytics

# Load pretrained model
model = YOLO('yolo11n.pt')

# Train
model.train(
    data="data.yaml",   # Dataset configuration file (contains train/val paths and class names)
    epochs=30,                  # Number of training epochs
    imgsz=1242,                 # Input image size (default is 640); larger values may improve accuracy
    batch=4,                    # Batch size; reduce if GPU memory is limited
    device='cpu',              # GPU device IDs to use; use 'cpu' for CPU training
    workers=4,                  # Number of dataloader workers (depends on CPU cores)
    amp=True,                   # Enable Automatic Mixed Precision (faster and uses less GPU memory)
    multi_scale=True            # Enable multi-scale training for better generalization
)