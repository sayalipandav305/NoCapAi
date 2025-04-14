import os, shutil, random

# This function splits and copies images to train/val folders
def split_dataset(base_dir, train_dir, val_dir, split_ratio=0.8):
    classes = ['real', 'fake']  # the two categories

    for cls in classes:
        src_dir = os.path.join(base_dir, cls)  # e.g. dataset/real/
        images = os.listdir(src_dir)
        random.shuffle(images)

        # Split based on ratio
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create target directories
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        # Copy training images
        for img in train_images:
            shutil.copy(os.path.join(src_dir, img), os.path.join(train_dir, cls, img))

        # Copy validation images
        for img in val_images:
            shutil.copy(os.path.join(src_dir, img), os.path.join(val_dir, cls, img))

    print("✅ Dataset split complete.")

# Set paths
base_dataset = 'deepfake_final/dataset'  # Folder with 'real' and 'fake'
train_path = 'deepfake_final/dataset/train'
val_path = 'deepfake_final/dataset/val'

# Run the function
split_dataset(base_dataset, train_path, val_path)
