import tensorflow as tf
from tensorflow import keras
import numpy as np

BW = 20  # 带宽
NFFT = int(BW * 3.2)
GESTURE_NUM = 3  # 手势数
TRAIN_SIZE = 150  # 训练集大小
PCAP_SIZE = 200  # 每个pcap包含的CSI数组个数

train_data = np.zeros((TRAIN_SIZE, PCAP_SIZE, NFFT))
train_result = np.zeros(TRAIN_SIZE)


for i in range(1, 51):
    # 读取npy文件
    temp_clap = np.load('data/clap' + str(i) + '.npy')
    temp_gesture = np.load('data/circle' + str(i) + '.npy')
    temp_result = np.load('data/static' + str(i) + '.npy')

    # CSI求模&&毛刺处理
    for j in range(PCAP_SIZE):
        for k in range(NFFT):
            # print(str(i) + ',' + str(j) + ',' + str(k))
            temp_clap[j][k] = abs(temp_clap[j][k])
            temp_gesture[j][k] = abs(temp_gesture[j][k])
            temp_result[j][k] = abs(temp_result[j][k])
            if k == 0 or k == 29 or k == 30 or k == 31 or k == 32 or k == 33 or k == 34 or k == 35:
                temp_clap[j][k] = 0
                temp_gesture[j][k] = 0
                temp_result[j][k] = 0

    # 加入学习队列
    temp_clap = temp_clap.astype('float64')
    temp_gesture = temp_gesture.astype('float64')
    temp_result = temp_result.astype('float64')
    train_data[i - 1] = temp_clap[0:PCAP_SIZE, :]
    train_data[50 + i - 1] = temp_gesture[0:PCAP_SIZE, :]
    train_data[100 + i - 1] = temp_result[0:PCAP_SIZE, :]
    train_result[i - 1] = 0
    train_result[50 + i - 1] = 1
    train_result[100 + i - 1] = 2

# 训练数据归一化
cnt = 0
for i in range(TRAIN_SIZE):
    for j in range(PCAP_SIZE):
        #print(train_data[i][j])
        CSI_max = train_data[i][j].max()
        if CSI_max == 0:
            cnt = cnt + 1
        for k in range(NFFT):
            # if train_data[i][j][k] == 0:
            #     train_data[i][j][k] = 1
            train_data[i][j][k] = train_data[i][j][k]/CSI_max
print(str(cnt))

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(PCAP_SIZE, NFFT)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(GESTURE_NUM)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_data, train_result)
