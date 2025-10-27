# Fine-tuning YOLOv11 on a Custom Dataset

## Step 1. Install Ultralytics

1. Create virtualEnv with conda:

    ``conda create -n yolov11 python=3.10`` 
2. Activate virtuelEnv:

    ```conda activate yolov11```

3. Install ultralytics:

    ```pip install ultralytics```

## Step 2. Prepare custom dataset
1. Annotation format:

    ```0 0.512 0.433 0.215 0.198``` -> ```Cls_id x_center, y_center, w, h```
    ```
    x_center = ((x_min + x_max) / 2) / W
    y_center = ((y_min + y_max) / 2) / H
    width    = (x_max - x_min) / W
    height   = (y_max - y_min) / H
    ```
2. Data structure:
    ```
    dataset/
    │
    ├── data.yaml               # Configuration file for training
    │
    ├── train/                  # Training set
    │   ├── images/                 # Training images
    │   │   ├── 0001.jpg
    │   │   ├── 0002.jpg
    │   │   └── ...
    │   │
    │   └── labels/                 # Training labels
    │       ├── 0001.txt
    │       ├── 0002.txt
    │       └── ...
    │
    └── val/                      # Validation set
        ├── images/                 # Validation images
        │   ├── 1001.jpg
        │   ├── 1002.jpg
        │   └── ...
        │
        └── labels/                 # Validation labels  
            ├── 1001.txt
            ├── 1002.txt
            └── ...
    ```
Data example:
```
In the path: data/KITTI-YOLO
```
# Step 3. Train the model 

1. Train the model:

    ``python train.py``

2. Use the pretrained model on videos

    ```python run.py```