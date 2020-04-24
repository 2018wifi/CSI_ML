from sklearn.cluster import KMeans
import numpy as np
from parameters import *


data = np.zeros((T_NUM*T_SIZE, PCAP_SIZE, NFFT))
for i in range(T_NUM):
    for j in range(T_SIZE):
        file = 'data'+'/T'+str(i+1)+'/T'+str(i+1)+'_'+str(j+1)+'.npy'
        temp = np.load(file)
        for k in range(PCAP_SIZE):
            temp[k] = abs(temp[k])
        temp.dtype = 'float64'
        temp = temp[0:PCAP_SIZE]
        # clean and normalize data
        for k in [0, 29, 30, 31, 32, 33, 34, 35]:
            temp[:, k] = 0

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(data)
print(kmeans.cluster_centers_)
kmeans.predict(data[0], data[50])


