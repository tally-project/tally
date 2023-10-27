import tensorflow as tf
import numpy as np
import random
import sys

import numpy
numpy.set_printoptions(threshold=sys.maxsize)

print("TensorFlow version:", tf.__version__)

tf.random.set_seed(12)
np.random.seed(12)
random.seed(12)

t = tf.random.uniform(shape=(1, 784), minval=0, maxval=1, dtype=tf.float32)
model = tf.keras.layers.Dense(128, activation='relu', bias_initializer='zeros')

pred = model(t)
print(pred)