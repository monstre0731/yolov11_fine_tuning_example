import os
import random
import shutil

# === Path configuration ===
src_img_dir = "data/KITTI/training/image_2"
src_label_dir = "data/KITTI/training/label_2"
dst_root = "data/KITTI-YOLO"
train_num, val_num = 200, 30

# === YOLO class mapping ===
classes = ["Car", "Pedestrian", "Cyclist"]

# === Create output directories in the form: dst_root/<split>/{images,labels} ===
for split in ["train", "val"]:
    os.makedirs(os.path.join(dst_root, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, split, "labels"), exist_ok=True)

# === Collect all image files ===
all_imgs = sorted([f for f in os.listdir(src_img_dir) if f.endswith(".png")])
random.seed(42)
random.shuffle(all_imgs)

train_imgs = all_imgs[:train_num]
val_imgs = all_imgs[train_num:train_num + val_num]


def convert_kitti_to_yolo(kitti_path, img_w=1242, img_h=375):
    """Convert a single KITTI label_2 file to YOLO format."""
    lines = []
    if not os.path.exists(kitti_path):
        return ""  # no label file
    with open(kitti_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            # KITTI bbox fields: parts[4]=xmin, [5]=ymin, [6]=xmax, [7]=ymax
            cls = parts[0]
            if cls not in classes:
                continue
            try:
                x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            except ValueError:
                continue
            cls_id = classes.index(cls)
            x_center = (x1 + x2) / 2.0 / img_w
            y_center = (y1 + y2) / 2.0 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def copy_and_convert(split_imgs, split_name):
    """Copy images and convert corresponding labels into dst_root/<split_name>/..."""
    copied = 0
    label_count = 0
    for img_name in split_imgs:
        base = os.path.splitext(img_name)[0]
        src_img = os.path.join(src_img_dir, img_name)
        src_label = os.path.join(src_label_dir, base + ".txt")

        dst_img = os.path.join(dst_root, split_name, "images", img_name)
        dst_label = os.path.join(dst_root, split_name, "labels", base + ".txt")

        # Copy image (overwrite if exists)
        shutil.copy(src_img, dst_img)
        copied += 1

        # Convert label and save (write empty file if no objects / no label file)
        yolo_txt = convert_kitti_to_yolo(src_label)
        with open(dst_label, "w") as f:
            if yolo_txt:
                f.write(yolo_txt + "\n")
                label_count += 1
            else:
                f.write("")  # empty label file for YOLO compatibility
    return copied, label_count


# === Run the conversion ===
train_copied, train_labels = copy_and_convert(train_imgs, "train")
val_copied, val_labels = copy_and_convert(val_imgs, "val")

print(f"✅ Done! Created YOLO dataset at: {dst_root}")
print(f"Train: {train_copied} images, {train_labels} labels with objects")
print(f"Val:   {val_copied} images, {val_labels} labels with objects")