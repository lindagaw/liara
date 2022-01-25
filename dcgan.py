import os
import numpy as np
import pretty_errors
import tensorflow as tf
import time
from office_31 import office_31_subset
from dcgan_generator import make_generator_model, generator_loss
from dcgan_discriminator import make_discriminator_model, discriminator_loss
import PIL
EPOCHS = 200
BATCH_SIZE = 320
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

amazon_xs, amazon_ys = office_31_subset('amazon')
#dslr_xs, dlsr_ys = office_31_subset('dslr')
#webcam_xs, webcam_ys = office_31_subset('webcam')

#noise = tf.random.normal([1, 100])

generator = make_generator_model()
discriminator = make_discriminator_model()

G = make_generator_model()
disc_G = make_discriminator_model()
F = make_generator_model()
disc_F = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

G_optimizer = tf.keras.optimizers.Adam(1e-5)
disc_G_optimizer = tf.keras.optimizers.Adam(1e-5)
F_optimizer = tf.keras.optimizers.Adam(1e-5)
disc_F_optimizer = tf.keras.optimizers.Adam(1e-5)

def cycle_loss(translated, second, translated_back, first):
    a = np.linalg.norm(translated-second)
    b = np.linalg.norm(translated_back-first)
    return a+b

def train_step(images):
    #noise = tf.random.normal([BATCH_SIZE, noise_dim])
    noise = amazon_xs

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, \
        tf.GradientTape() as F_tape, tf.GradientTape() as disc_F_tape, \
        tf.GradientTape() as G_tape, tf.GradientTape() as disc_G_tape:

        translated = G(noise, training=True)
        real_output = disc_G(noise, training=True)
        fake_output = disc_G(translated, training=True)
        gen_G_loss = generator_loss(fake_output)
        disc_G_loss = discriminator_loss(real_output, fake_output)


        translated_back = F(translated, train=True)
        real_output = disc_F(images, training=True)
        fake_output = disc_F(translated_back, training=True)
        gen_F_loss = generator_loss(fake_output)
        disc_F_loss = discriminator_loss(real_output, fake_output)

        c_loss = cycle_loss(translated, images, translated_back, noise)


    gradients_of_F = F_tape.gradient(gen_F_loss+c_loss, F.trainable_variables)
    gradients_of_disc_F = disc_F_tape.gradient(disc_F_loss, disc_F.trainable_variables)
    gradients_of_G = G_tape.gradient(gen_G_loss+c_loss, G.trainable_variables)
    gradients_of_disc_G = disc_G_tape.gradient(disc_G_loss, disc_G.trainable_variables)


    G_optimizer.apply_gradients(zip(gradients_of_G , G.trainable_variables))
    disc_G_optimizer.apply_gradients(zip(gradients_of_disc_G, disc_G.trainable_variables))
    F_optimizer.apply_gradients(zip(gradients_of_F , F.trainable_variables))
    disc_F_optimizer.apply_gradients(zip(gradients_of_disc_F, disc_F.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

train(np.asarray([amazon_xs]), EPOCHS)

data = PIL.Image.fromarray(amazon_xs[0], 'RGB')
data.save('input.png')

generated = np.squeeze(G(amazon_xs[0]))

print(generated.shape)

data = PIL.Image.fromarray(generated, 'RGB')
data.save('output.png')
