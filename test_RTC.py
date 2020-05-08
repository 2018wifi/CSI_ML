from real_time_cluster import Cluster
from parameters import *
import numpy as np
import torch
from CNN_pytorch import device
import threading

def test():
    for i in range(T_NUM):
        for j in range(T_SIZE):
            file = 'data/T' + str(i + 1) + '/T' + str(i + 1) + '_' + str(j + 1) + '.npy'
            temp_x = np.load(file)
            for k in range(PCAP_SIZE):
                temp_x[k] = abs(temp_x[k])
            temp_x = torch.from_numpy(temp_x.astype('float'))[0:PCAP_SIZE]
            # clean and normalize data
            for k in [0, 29, 30, 31, 32, 33, 34, 35]:
                temp_x[:, k] = 0
            for k in range(temp_x.shape[0]):
                CSI_max = temp_x[k].max()
                for p in range(temp_x.shape[1]):
                    temp_x[k][p] = temp_x[k][p] / CSI_max
            temp_x = torch.reshape(temp_x, (1, 1, PCAP_SIZE, NFFT))
            temp_x = temp_x.to(device)
            temp_x = temp_x.float()
            cluster.put_data(temp_x)
            # print(len(cluster.pool))

if __name__ == '__main__':
    path = 'D:\\Users\\SOBER\\PycharmProjects\\CSI_ML\\model.pkl'
    cluster = Cluster(path)
    cluster.run()
    t = threading.Thread(target=test())
    t.start()
    print('cool')
