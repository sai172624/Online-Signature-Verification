
import pandas as pd
import numpy as np
from PIL import Image
from keras import backend as K
from keras import layers, models
from keras.layers import Dense, ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np

import matplotlib.pyplot as plt

def prediction(img1,img2):
    output1, output2 = model([img1, img2], training=False)
    euclidean_distance = tf.norm(output1 - output2, axis=1)
    if tf.reduce_all(tf.equal(label, list_0)):
        label_text = "Original"
    else:
        label_text = "Forged"

    concatenated = tf.concat([x0, x1], axis=1)
    concatenated = concatenated.numpy().squeeze()

    plt.imshow(concatenated, cmap='gray')
    plt.title(f'Dissimilarity: {euclidean_distance.numpy()[0]:.2f} Label: {label_text}')
    plt.show()

    