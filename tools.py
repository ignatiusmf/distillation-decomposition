import os
import shutil

val_dir = "./data/tiny-imagenet-200/val"
images_dir = os.path.join(val_dir, "images")
annotations_file = os.path.join(val_dir, "val_annotations.txt")

# Parse val_annotations.txt
with open(annotations_file, 'r') as f:
    lines = f.readlines()

# Create subdirectories and move images
for line in lines:
    parts = line.strip().split('\t')
    image_filename = parts[0]
    class_label = parts[1]

    class_dir = os.path.join(images_dir, class_label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    src_path = os.path.join(images_dir, image_filename)
    dst_path = os.path.join(class_dir, image_filename)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)