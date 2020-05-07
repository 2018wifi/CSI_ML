import threading
import torch
from collections import deque
import socket
import struct
import numpy as np
import datetime
import math
from parameters import *

BW = 20
NFFT = int(BW * 3.2)
PORT = 3600                      # 传输端口
TIMEMAX = 10                      # 程序运行时间最大值（单位：秒）


def tcpip():
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.bind(("", PORT))
    tcp_socket.listen(128)
    reply = "Received"
    print("Waiting for connection...")
    client_socket, client_addr = tcp_socket.accept()

    time_start = datetime.datetime.now()
    while True:
        data = np.zeros((PCAP_SIZE, NFFT))
        for i in range(PCAP_SIZE):
            print("Listening...")
            buffer = client_socket.recv(1024)
            if len(buffer) != 1024:
                continue
            time_now = datetime.datetime.now() - time_start  # 时间差
            time_now = time_now.total_seconds()
            client_socket.send(reply.encode())
            data = parse(buffer)
            csi = read_csi(data)
            data[i] = csi
        data = torch.from_numpy(data.astype('float64'))
        cluster.put_data(data)


def parse(buffer):      # 解析二进制流
    nbyte = int(len(buffer))        # 字节数
    data = np.array(struct.unpack(nbyte * "B", buffer), dtype=np.uint8)

    return data


def read_csi(data):     # 提取CSI信息，并转换成矩阵
    csi = np.zeros(NFFT, dtype=np.float)
    sourceData = data[18:274]
    sourceData.dtype = np.int16
    csi_data = sourceData.reshape(-1, 2).tolist()
    i = 0
    for x in csi_data:
        csi[i] = math.sqrt(x[0]**2 + x[1]**2)
        i += 1
    for i in [0, 29, 30, 31, 32, 33, 34, 35]:
        csi[i] = 0
    max_data = max(csi)
    for i in range(NFFT):
        csi[i] = csi[i]/max_data
    return csi

# class SortedList:
#     def __init__(self):
#         self.data = []
#
#     def take_second(self, elem):
#         return elem[1]
#
#     def push(self, data):
#         self.data.append(data)
#         self.data.sort(key=self.take_second(), reverse=True)
#
#     def get(self):
#         return self.data[len(self.data)]


class Cluster:
    def __init__(self, model_path, num=2):
        self.pool = []
        self.queue = deque()
        self.model = torch.load(model_path)
        self.lock = threading.Lock()

    # You can use the "put_data" method anytime, anywhere
    def put_data(self, data):
        # 'data' should be a tuple (A, B, C)
        # B is a time stamp, which will be used to match different data sent by different rasPi in the same time
        # A is an integer indicating which rasPi the csi data belongs to,
        # B is a package containing N pieces of CSI data, so its size should be (N, NFFT)
        # In addition, N should be same as the data used in training the model
        with self.lock:
            self.pool.append(data)

    def cluster(self):
        while True:
            try:
                with self.lock:
                    data = self.pool.popleft()
                y = self.model(data)
                print(y)
            except:
                pass

    def run(self):
        t = threading.Thread(target=self.cluster)
        t.start()


if __name__ == '__main__':
    path = 'model.pkl'
    cluster = Cluster(path, 2)
    cluster.run()

