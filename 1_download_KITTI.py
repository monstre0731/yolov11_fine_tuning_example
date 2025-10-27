import os
import random
import urllib.request
import zipfile
import shutil

# ===== Data download function =====
def download_kitti(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print(f"{filename} already exists.")

# ===== Unzip files =====
def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

# ===== Convert KITTI to YOLO =====
def convert_kitti_to_yolo(kitti_label_path, output_path, image_width=1242, image_height=375):
    class_map = {
        "Car": 0,
        "Van": 1,
        "Truck": 2,
        "Pedestrian": 3,
        "Person_sitting": 4,
        "Cyclist": 5,
        "Tram": 6,
        "Misc": 7,
        "DontCare": -1
    }

    with open(kitti_label_path, "r") as f:
        lines = f.readlines()

    yolo_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        cls_name = parts[0]
        if cls_name not in class_map or class_map[cls_name] == -1:
            continue
        cls_id = class_map[cls_name]
        xmin, ymin, xmax, ymax = map(float, parts[4:8])
        x_center = ((xmin + xmax) / 2.0) / image_width
        y_center = ((ymin + ymax) / 2.0) / image_height
        w = (xmax - xmin) / image_width
        h = (ymax - ymin) / image_height
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    with open(output_path, "w") as f:
        f.writelines(yolo_lines)

# ===== Main  =====
def create_kitti_yolo_demo():
    os.makedirs("kitti_yolo_demo", exist_ok=True)

    # Download images and labels (About 13 GB)
    urls = {
        "images": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "labels": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    }
    download_kitti(urls["images"], "data_object_image_2.zip")
    download_kitti(urls["labels"], "data_object_label_2.zip")

    # Unzip
    extract_zip("data_object_image_2.zip", "./data/KITTI")
    extract_zip("data_object_label_2.zip", "./data/KITTI")


# ===== 运行 =====
if __name__ == "__main__":
    create_kitti_yolo_demo()