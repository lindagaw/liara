import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time
from office_31 import office_31_subset
from dcgan_generator import make_generator_model, generator_loss
from dcgan_discriminator import make_discriminator_model, discriminator_loss

amazon_xs, amazon_ys = office_31_subset('amazon')
dslr_xs, dlsr_ys = office_31_subset('dslr')
webcam_xs, webcam_ys = office_31_subset('webcam')


resnet_model = tf.keras.applications.resnet50.ResNet50(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(32, 32, 3), pooling=None, classes=1000
)


results = resnet_model.predict(amazon_xs)

print(results.shape)
