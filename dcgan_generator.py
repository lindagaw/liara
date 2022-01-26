import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import tensorflow as tf

from IPython import display

from tensorflow.keras import Sequential
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, LeakyReLU, Reshape
from tensorflow.keras.optimizers import SGD

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def make_generator_model():
    model = Sequential()
    model.add(Dense(4 * 4 * 512, input_dim=100,
                                   kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('relu'))
    model.add(Reshape((4, 4, 512)))

    model.add(Conv2DTranspose(256, 3, strides=2, padding='same',
                                             kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(128, 3, strides=2, padding='same',
                                             kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(64, 3, strides=2, padding='same',
                                             kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('relu'))

    model.add(Conv2D(3, 3, padding='same',
                                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
    model.add(Activation('tanh'))

    return model

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
