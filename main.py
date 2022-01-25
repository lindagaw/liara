import tensorflow as tf
from tensorflow.keras import layers

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

x = amazon_xs

inputs = tf.keras.Input(shape=(224, 224, 3))
x = resnet_model(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = layers.Dense(31)(x)
model = tf.keras.Model(inputs, outputs)

results = model.predict(amazon_xs)

print(results.shape)
