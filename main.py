import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import pretty_errors

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import time
from office_31 import office_31_subset
from dcgan_generator import make_generator_model, generator_loss
from dcgan_discriminator import make_discriminator_model, discriminator_loss

amazon_xs, amazon_ys = office_31_subset('amazon')
dslr_xs, dlsr_ys = office_31_subset('dslr')
webcam_xs, webcam_ys = office_31_subset('webcam')

def mil_squared_error(y_true, y_pred):
    return tf.keras.backend.square(tf.keras.backend.max(y_pred) - tf.keras.backend.max(y_true))


def tuned_resnet(X, y):
    resnet_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=(128, 128, 3), pooling=None)

    flattened = tf.keras.layers.Flatten()(resnet_model.output)
    dense = tf.keras.layers.Dense(31, activation='softmax', name="AddedDense2")(flattened)
    model = tf.keras.models.Model(inputs=resnet_model.input, outputs=dense)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-6),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.CategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[tf.keras.metrics.CategoricalAccuracy(), mil_squared_error],
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
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    print('the testing accuracy is {} '.format(acc))

X = amazon_xs
y = tf.keras.utils.to_categorical(amazon_ys)
tuned_resnet(X, y)
