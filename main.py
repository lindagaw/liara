import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import time
from office_31 import office_31_subset
from dcgan_generator import make_generator_model, generator_loss
from dcgan_discriminator import make_discriminator_model, discriminator_loss

amazon_xs, amazon_ys = office_31_subset('amazon')
dslr_xs, dlsr_ys = office_31_subset('dslr')
webcam_xs, webcam_ys = office_31_subset('webcam')

X = amazon_xs
y = tf.keras.utils.to_categorical(amazon_ys)

resnet_model = tf.keras.applications.resnet50.ResNet50(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(32, 32, 3), pooling=None)

flattened = tf.keras.layers.Flatten()(resnet_model.output)
dense = tf.keras.layers.Dense(31, activation='softmax', name="AddedDense2")(flattened)
model = tf.keras.models.Model(inputs=resnet_model.input, outputs=dense)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # Optimizer
    # Loss function to minimize
    loss=tf.keras.losses.CategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=200,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_val, y_val),
)

y_pred = model.predict(X_test)
acc = accuracy_score(np.squeeze(y_test), np.squeeze(y_pred))

print(acc)
