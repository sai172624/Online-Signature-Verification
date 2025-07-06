import pandas as pd
import numpy as np
import os
from PIL import Image
from keras import backend as K
from keras.layers import Dense, ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.models import load_model


def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

class Config():
    training_dir ="C:\\Users\\pshab\\Desktop\\signature verification\\signver\\archive\\sign_data\\sign_data\\train"
    testing_dir = "C:\\Users\\pshab\\Desktop\\signature verification\\signver\\archive\\sign_data\\sign_data\\test"
    train_batch_size = 16
    train_number_epochs =10

training_dir="C:\\Users\\pshab\\Desktop\\signature verification\\signver\\archive\\sign_data\\sign_data\\train"
training_csv="C:\\Users\\pshab\\Desktop\\signature verification\\signver\\archive\\sign_data\\sign_data\\train_data.csv"
testing_csv="C:\\Users\\pshab\\Desktop\\signature verification\\signver\\archive\\sign_data\\sign_data\\test_data.csv"
testing_dir="C:\\Users\\pshab\\Desktop\\signature verification\\signver\\archive\\sign_data\\sign_data\\test"

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

def dataset_generator_function(training_csv, training_dir, transform=None):
    df = pd.read_csv(training_csv)
    df.columns = ["image1", "image2", "label"]

    for _, row in df.iterrows():
        image1_path = os.path.join(training_dir, row['image1'])
        image2_path = os.path.join(training_dir, row['image2'])

        # Open images
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)

        img0 = transform(img0)
        img1 = transform(img1)
        # Ensure images are of shape (224, 224, 3)
        img0 = np.array(img0)
        img1 = np.array(img1)
        
        # If image shape is not as expected, resize and normalize
        if img0.shape != (224, 224, 3):
            img0 = np.resize(img0, (224, 224, 3))
        if img1.shape != (224, 224, 3):
            img1 = np.resize(img1, (224, 224, 3))

        label = float(row['label'])  # Yield as a scalar

        yield img0, img1, label

# Define transformations
def transform(image):
    image = image.resize((224, 224))  # Resize image
    image = image.convert("RGB")
    image = np.array(image) / 255.0  # Normalize image
    return image
# Load the full dataset
full_dataset = tf.data.Dataset.from_generator(
    lambda: dataset_generator_function(training_csv=training_csv, training_dir=training_dir, transform=transform),
    output_signature=(
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),  # Scalar label
    )
)

train_dataset = full_dataset.take(10000)
train_dataset = train_dataset.batch(Config.train_batch_size).prefetch(tf.data.AUTOTUNE)

@register_keras_serializable()  # Ensures that Keras can serialize this model
class SiameseNetwork(tf.keras.Model):
    def __init__(self,**kwargs):
        super(SiameseNetwork, self).__init__(**kwargs)
        self.cnn1 = models.Sequential([
            layers.Conv2D(96, kernel_size=7, strides=1, padding='same',activation='relu', input_shape=(224,224, 3)),
            layers.MaxPooling2D(pool_size=3, strides=2),
            layers.Conv2D(256, kernel_size=5, strides=1, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=3, strides=2),
            layers.Dropout(0.3),
            layers.Conv2D(384, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=3, strides=2),
            layers.Dropout(0.3),
        ])
        self.flatten = layers.Flatten()
        self.fc1 = models.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu')
        ])

    def call(self, inputs, training=False):
        img1, img2 = inputs
        output1 = self.cnn1(img1)
        output1 = self.flatten(output1)
        output1 = self.fc1(output1)

        output2 = self.cnn1(img2)
        output2 = self.flatten(output2)
        output2 = self.fc1(output2)

        return output1, output2

    def get_config(self):
        # Return any configuration needed to reinstantiate the model
        config = super(SiameseNetwork, self).get_config()
        # Add any custom layers or parameters here, if needed
        return config

    @classmethod
    def from_config(cls, config):
        # Pop 'trainable' or any other kwargs not needed
        config.pop('trainable', None)
        config.pop('dtype', None)
        return cls(**config)


class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def call(self, outputs, labels):
        output1, output2 = outputs
        labels = tf.cast(labels, tf.float32)
        euclidean_distance = tf.norm(output1 - output2, axis=1)
        loss_contrastive = tf.reduce_mean((1 - labels) * tf.square(euclidean_distance) +
                                          labels * tf.square(tf.maximum(self.margin - euclidean_distance, 0.0)))
        return loss_contrastive


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.99, beta_2=0.999, epsilon=1e-8, decay=0.0005)
criterion = ContrastiveLoss()
model=load_model('C:\\Users\\pshab\\Desktop\\signature verification\\signver\\backend\\mymodel.keras', custom_objects={'SiameseNetwork': SiameseNetwork})

# Updated prediction function with optimal threshold
def prediction(img1, img2):
    output1, output2 = model([img1, img2], training=False)
    euclidean_distance = tf.norm(output1 - output2, axis=1)
    euclidean_distance = euclidean_distance.numpy() if tf.executing_eagerly() else euclidean_distance

    if euclidean_distance <0.2445 :
        label_text = "Original"
    else:
        label_text = "Forged"

    return label_text, euclidean_distance


