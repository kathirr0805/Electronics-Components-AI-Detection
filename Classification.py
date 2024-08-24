import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Paths
dataset_dir = 'F:/Projects/AI/Electro AI/Datasets/archive/images'
model_path = os.path.join(dataset_dir, 'model.h5')

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(img_path, target_size):
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"The image file does not exist: {img_path}")
        
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Scale pixel values to [0, 1]
    return img_array

# Load class names (should match the order used during training)
train_dir = os.path.join(dataset_dir, 'train')
class_names = sorted(os.listdir(train_dir))

# Create a Tkinter root window (it will not be shown)
root = Tk()
root.withdraw()  # Hide the root window

# Open file dialog to select an image file
img_path = askopenfilename(
    title="Select an Image File",
    filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
)

if not img_path:
    print("No file selected.")
else:
    target_size = (150, 150)  # Replace with the size used during training

    # Print the path to verify
    print(f"Image path: {img_path}")

    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(img_path, target_size)

        # Make a prediction
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = class_names[predicted_class[0]]

        # Display the result
        print(f"Predicted class: {predicted_label}")

        # Display and save the image
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(f"Predicted class: {predicted_label}")
        plt.axis('off')

        # Save the image with a new name indicating the predicted class
        save_path = os.path.join(os.path.dirname(img_path), f"{predicted_label}_{os.path.basename(img_path)}")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved the image with prediction to {save_path}")

        plt.show()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
