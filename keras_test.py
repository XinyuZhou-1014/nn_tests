import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Reshape, Activation
import tensorflow as tf
import numpy as np

model = Sequential()
model.add(Reshape([28, 28, 1], input_shape=(784, )))
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 padding="same",
                 activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 padding="same",
                 activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Reshape([7 * 7 * 64]))
model.add(Dense(1024, activation="relu"))
model.add(Dense(10))
model.compile(
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"])
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

one_hot_labels = keras.utils.to_categorical(train_labels, num_classes=10)
one_hot_eval_labels = keras.utils.to_categorical(eval_labels, num_classes=10)


model.fit(train_data, one_hot_labels, epochs=10, batch_size=32,
          validation_data=[eval_data, one_hot_eval_labels])
