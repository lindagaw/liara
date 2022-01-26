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

from tf.keras import Sequential
from tf.keras.initializers import TruncatedNormal
from tf.keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, LeakyReLU, Reshape
from tf.keras.optimizers import SGD

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def make_discriminator_model():
    model = Sequential()
    model.add(Conv2D(128, 3, strides=2, input_shape=(32, 32, 3), padding='same',
                                        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, 3, strides=2, padding='same',
                                        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
    self.generator_model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(512, 3, strides=2, padding='same',
                                        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
    self.generator_model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1024, 3, strides=2, padding='same',
                                        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
    self.generator_model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(1, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
    self.generator_model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('sigmoid'))

    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
