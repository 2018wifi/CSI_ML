import dpkt
import time
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

BW = 20
NFFT = int(BW * 3.2)
t_c = 3
rate = 50  # 发包速率，单位：包/秒

class Pcap:
    def __init__(self, file_name):
        self.pcap_name = file_name

    def __parse_udp(self, buffer):      # 解析udp包的二进制流
        nbyte = int(len(buffer))        # 字节数
        data = np.array(struct.unpack(nbyte * "B", buffer), dtype=np.uint8)
        return data

    def __read_header(self, data):  # 提取头信息
        header = {}

        header["magic_bytes"] = data[:4]
        header["source_mac"] = data[4:10]
        header["sequence_number"] = data[10:12]

        coreSpatialBytes = int.from_bytes(data[12:14], byteorder="little")
        header["core"] = [int(coreSpatialBytes & x != 0) for x in range(3)]
        header["spatial_stream"] = [int(coreSpatialBytes & x != 0) for x in range(3, 6)]

        header["channel_spec"] = data[14:16]
        header["chip"] = data[16:18]

        return header

    def __read_csi(self, data):     # 提取CSI信息，并转换成矩阵
        csi = np.zeros(NFFT, dtype=np.complex)
        sourceData = data[18:]
        sourceData.dtype = np.int16
        csi_data = sourceData.reshape(-1, 2).tolist()
        i = 0
        for x in csi_data:
            csi[i] = np.complex(x[0], x[1])
            i += 1
        return csi

    def parse(self):                                            # 提取CSI信息和时间戳信息，并转换成矩阵
        f = open("data_raw/" + self.pcap_name + ".pcap", 'rb')
        pcap = dpkt.pcap.Reader(f)

        csi_matrix_list = []
        ts_raw_list = []
        ts_list = []
        t_d = []
        no = 0

        for ts, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
            transf_data = ip.data
            payload = transf_data.data      # 逐层解包

            if len(payload) != 274:         # 大小不对时舍弃这个包
                print('*')
                continue

            data = self.__parse_udp(payload)  # 处理数据
            csi = self.__read_csi(data)

            if no == 0:
                ts_s = ts

            tt = int(time.strftime("%S",time.localtime(ts - ts_s)))
            if tt < t_c:
                csi_matrix_list.append(csi)
                ts_list.append(ts - ts_s)
                ts_raw_list.append(tt)      # 用于检验原始数据使用
                if no > 0:
                    t_d.append(-ts_list[no] + ts_list[no - 1])      #负差值，方便后面从大到小排序
            no = no + 1

        ts_lack = t_c * rate - len(ts_list)

        td_matrix = np.array(t_d)
        td_sort = td_matrix.argsort()

        for i in range(ts_lack):
            shift = 0
            ma_i = [0 for j in range(NFFT)]

            for j in range(i):
                if td_sort[i] > td_sort[j]:
                    shift = shift + 1

            index = td_sort[i] + shift

            for j in range(NFFT):                       # 取索引位前后的中值作为插值
                ma_i[j] = (abs(csi_matrix_list[index][j]) + abs(csi_matrix_list[index + 1][j])) / 2
            csi_matrix_list.insert(index + 1, ma_i)

        self.csi_matrix = np.array(csi_matrix_list)
        self.ts_raw_matrix = np.array(ts_raw_list)
        f.close()

    def save(self):                                     # 写入npy文件
        with open("data/" + self.pcap_name + ".npy", 'wb'):
            pass
        np.save("data/" + self.pcap_name + ".npy", self.csi_matrix)

    def draw(self):                    # 绘制未插值的pcap数据包接收情况
        ts_matrix = self.ts_raw_matrix
        # ts_unique = np.unique(ts_matrix)
        # xnum = len(ts_unique)
        # no = [i for i in range(xnum + 1)]
        # x = [(str(no[i]) + '-' + str(no[i + 1])) for i in range(xnum)]
        xnum = ts_matrix[len(ts_matrix) - 1]
        x = [i + 1 for i in range(xnum + 1)]
        data_count = []
        data = sorted(ts_matrix)
        for i in range(xnum + 1):
            # print(data.count(i))
            data_count.append(data.count(i))
        plt.ylabel('number of packet')
        plt.xlabel('time')
        # plt.ylim(0, max(data_count) + 1)
        # plt.xlim(0, max(x) + 1)
        width = 0.6
        x_major_locator = MultipleLocator(2)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        bn = plt.bar(x, data_count, width)  # 初始状态的图
        plt.tick_params(axis='x', labelsize=5)
        # plt.xticks(rotation=-45)
        ax.set_title(self.pcap_name)
        for b in bn:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),b.get_height(), fontsize=7,ha = 'center',va='bottom')
        plt.show()

if __name__ == '__main__':
    for i in range(1, 71):
        pcap_name = "T5/T5_" + str(i)
        pcap = Pcap(pcap_name)
        pcap.parse()
        pcap.save()
        # pcap.draw()

