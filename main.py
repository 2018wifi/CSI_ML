import tensorflow as tf
from tensorflow import keras
import numpy as np

BW = 20  # 带宽
NFFT = int(BW * 3.2)
GESTURE_NUM = 3  # 手势数
TRAIN_SIZE = 45  # 训练集大小
PCAP_SIZE = 10  # 每个pcap包含的CSI数组个数

train_data = np.zeros((TRAIN_SIZE, PCAP_SIZE, NFFT))
train_result = np.zeros(TRAIN_SIZE)


for i in range(1, 16):
    # 读取npy文件
    temp_clap = np.load('data/clap' + str(i) + '.npy')
    temp_gesture = np.load('data/gesture' + str(i) + '.npy')
    temp_result = np.load('data/result' + str(i) + '.npy')

    # CSI求模&&毛刺处理
    for j in range(PCAP_SIZE):
        for k in range(NFFT):
            temp_clap[j][k] = abs(temp_clap[j][k])
            temp_gesture[j][k] = abs(temp_gesture[j][k])
            temp_result[j][k] = abs(temp_result[j][k])
            if 29 <= abs(k - 32) <= 32 or k == 32:
                temp_clap[j][k] = 0
                temp_gesture[j][k] = 0
                temp_result[j][k] = 0

    # 加入学习队列
    temp_clap = temp_clap.astype('float64')
    temp_gesture = temp_gesture.astype('float64')
    temp_result = temp_result.astype('float64')
    train_data[i - 1] = temp_clap[0:PCAP_SIZE, :]
    train_data[2 * (i - 1)] = temp_gesture[0:PCAP_SIZE, :]
    train_data[3 * (i - 1)] = temp_result[0:PCAP_SIZE, :]
    train_result[i - 1] = 0
    train_result[2 * (i - 1)] = 1
    train_result[3 * (i - 1)] = 2


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(PCAP_SIZE, NFFT)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(GESTURE_NUM)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_data, train_result)
