import os
import numpy as np
import pretty_errors
import tensorflow as tf
import time
from office_31 import office_31_subset
from dcgan_generator import make_generator_model, generator_loss
from dcgan_discriminator import make_discriminator_model, discriminator_loss
import PIL
EPOCHS = 50
BATCH_SIZE = 32
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

amazon_xs, amazon_ys = office_31_subset('amazon')
#dslr_xs, dlsr_ys = office_31_subset('dslr')
#webcam_xs, webcam_ys = office_31_subset('webcam')

#noise = tf.random.normal([1, 100])

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

train(np.asarray([amazon_xs]), EPOCHS)

noise = tf.random.normal([1, 100])
generated = generator(noise)

print(generated.shape)

data = PIL.Image.fromarray(generated.numpy())
data.save('output.png')
