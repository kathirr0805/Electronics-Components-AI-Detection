import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
dataset_dir = 'F:/Projects/AI/Electro AI/Datasets/archive/images'
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')

# Create directories for training and validation data
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Function to create train and validation splits
def create_train_validation_split(dataset_dir, train_dir, validation_dir, test_size=0.2):
    # Iterate over each class directory in the dataset
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        
        # Skip non-directory files
        if not os.path.isdir(class_path):
            continue

        # Create corresponding directories in train and validation folders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)
        
        # Get all image files in the class directory
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        # Skip if no images are found
        if not images:
            print(f"No images found in {class_path}, skipping.")
            continue
        
        # Split images into train and validation sets
        train_images, validation_images = train_test_split(images, test_size=test_size, random_state=42)
        
        # Copy files to train and validation directories
        for image in train_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(train_dir, class_name, image))
        
        for image in validation_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(validation_dir, class_name, image))
        
        print(f"Processed class: {class_name}")

# Create train and validation sets
create_train_validation_split(dataset_dir, train_dir, validation_dir)

