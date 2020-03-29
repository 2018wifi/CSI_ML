import tensorflow as tf
from tensorflow import keras
import numpy as np
import pcap_reader
BW = 20  # 带宽
NFFT = int(BW * 3.2)
GESTURE_NUM = 3  # 手势数
TRAIN_SIZE = 10000  # 训练集大小
PCAP_SIZE = 1000  # 每个pcap包含的CSI数组个数

train_data = np.zeros((TRAIN_SIZE, PCAP_SIZE, NFFT))
train_result = np.zeros(TRAIN_SIZE)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(PCAP_SIZE, NFFT)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(GESTURE_NUM)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_data, train_result, epoches=10)

