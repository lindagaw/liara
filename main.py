import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from office_31 import office_31_subset
from dcgan_generator import make_generator_model, generator_loss
from dcgan_discriminator import make_discriminator_model, discriminator_loss

amazon_xs, amazon_ys = office_31_subset('amazon')
dslr_xs, dlsr_ys = office_31_subset('dslr')
webcam_xs, webcam_ys = office_31_subset('webcam')


resnet_model = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=(224, 224, 3), pooling=None, classes=1000
)

attached_model = tf.keras.Sequential()
attached_model.add(layers.Dense(31))

attached_optimizer = tf.keras.optimizers.Adam(1e-4)
resnet_optimizer = tf.keras.optimizers.Adam(1e-4)

def train_step(images):

    with tf.GradientTape() as resnet_tape, tf.GradientTape() as attached_tape:
        resnet_output = resnet_model(images, training=True)
        attached_output = attached_model(resnet_output, training=True)

        resnet_loss = generator_loss(fake_output)
        attached_loss = discriminator_loss(real_output, fake_output)

    gradients_of_resnet = resnet_tape.gradient(resnet_loss, resnet_model.trainable_variables)
    gradients_of_attached = attached_tape.gradient(attached_loss, attached_model.trainable_variables)

    resnet_optimizer.apply_gradients(zip(gradients_of_resnet, resnet_model.trainable_variables))
    attached_optimizer.apply_gradients(zip(gradients_of_attached, attached_model.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

EPOCHS = 10

train(np.asarray([amazon_xs]), EPOCHS)

results = model.predict(resnet_model.predict(amazon_xs))

print(results.shape)
