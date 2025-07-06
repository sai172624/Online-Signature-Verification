from PIL import Image
from train2 import prediction
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

def load_model_and_verify(image1_path,image2_path):

    img0 = Image.open(image1_path)
    img1 = Image.open(image2_path)
    img0 = transform(img0)
    img1 = transform(img1)

    img0 = np.array(img0)  # Convert PIL image to numpy array
    img1 = np.array(img1)  # Convert PIL image to numpy array

    # Ensure images are of shape (105, 105, 1)  # Add channel dimension
    img0 = np.resize(img0, (224, 224, 3))
    img0 = np.resize(img0, (224, 224, 3))
    img0 = np.expand_dims(img0, axis=0)  # Add batch dimension
    img1 = np.expand_dims(img1, axis=0)  # Add batch dimension

    predict,euclid=prediction(img0,img1)
    return predict,euclid

def transform(image):
    image = image.resize((224, 224))  # Resize image
    image=image.convert("RGB")
    image = np.array(image) / 255.0  # Normalize image
    return image