import os
import shutil
import random

# Paths
original_dataset = "G:\Floravision8\fullcleaned"   
output_dataset = "G:\Floravision8\dataset_split"

# Delete old split dataset (Solution 3)
if os.path.exists(output_dataset):
    shutil.rmtree(output_dataset)

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Ensure output folders exist
for split in ["train", "val", "test"]:
    split_path = os.path.join(output_dataset, split)
    os.makedirs(split_path, exist_ok=True)

# Loop through each class folder
for class_name in os.listdir(original_dataset):
    class_path = os.path.join(original_dataset, class_name)
    if not os.path.isdir(class_path):
        continue

    # Collect all images in this class
    images = os.listdir(class_path)
    random.shuffle(images)  # shuffle for randomness

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # Remaining goes to test
    n_test = n_total - n_train - n_val

    # Split dataset
    train_files = images[:n_train]
    val_files = images[n_train:n_train+n_val]
    test_files = images[n_train+n_val:]

    # Function to copy files
    def copy_files(file_list, split):
        split_class_path = os.path.join(output_dataset, split, class_name)
        os.makedirs(split_class_path, exist_ok=True)
        for file in file_list:
            src = os.path.join(class_path, file)
            dst = os.path.join(split_class_path, file)
            shutil.copy(src, dst)

    # Copy to respective folders
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print(f"✅ {class_name}: {n_train} train, {n_val} val, {n_test} test")

