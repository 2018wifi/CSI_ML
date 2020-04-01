import tensorflow as tf
from tensorflow import keras
import numpy as np

fake_data = np.zeros((1000000, 2))
train_data = np.random.randint(1, 9, size=(1000000, 2)).astype('float64')
train_answer = np.zeros(1000000)
for i in range(1000000):
    train_answer[i] = train_data[i][0] + train_data[i][1]
    train_data[i][0] = train_data[i][0]/10
    train_data[i][1] = train_data[i][1]/10
model = keras.Sequential([
    keras.layers.Dense(2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(19)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(fake_data, train_answer)
