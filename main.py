import os
import numpy as np
import pretty_errors
import tensorflow as tf

from office_31 import office_31_subset
from dcgan_generator import make_generator_model
from dcgan_discriminator import make_discriminator_model

office_31_subset('amazon')
office_31_subset('dslr')
office_31_subset('webcam')


generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)


discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)
