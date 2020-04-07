import dpkt
import time
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

BW = 20
NFFT = int(BW * 3.2)
pcap_name = 'test1-1'
t_c = 3
rate = 45  # 发包速率，单位：包/秒

def parse_udp(buffer):      # 解析udp包的二进制流
    nbyte = int(len(buffer))        # 字节数
    data = np.array(struct.unpack(nbyte * "B", buffer), dtype=np.uint8)

    return data

def read_header(data):  # 提取头信息
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

def read_csi(data):     # 提取CSI信息，并转换成矩阵
    csi = np.zeros(NFFT, dtype=np.complex)
    sourceData = data[18:]
    sourceData.dtype = np.int16
    csi_data = sourceData.reshape(-1, 2).tolist()

    i = 0
    for x in csi_data:
        csi[i] = np.complex(x[0], x[1])
        i += 1

    return csi

def parse_pcap(pcap_file):        # 将源pcap文件转为numpy矩阵的二进制文件
    f = open("data_raw/" + pcap_file + ".pcap", 'rb')
    pcap = dpkt.pcap.Reader(f)

    matrix_list = []
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

        if no == 0:
            ts_s = ts

        data = parse_udp(payload)
        csi = read_csi(data)


        # ts_list.append(time.strftime("%H:%M:%S",time.localtime(ts)))
        tt = int(time.strftime("%S",time.localtime(ts - ts_s)))
        if tt < t_c:
            matrix_list.append(csi)
            ts_list.append(ts - ts_s)
            # ts_list.append(tt)
            if no > 0:
                t_d.append(-ts_list[no] + ts_list[no - 1])      #负差值，方便后面从大到小排序
        no = no + 1

    ts_lack = t_c * rate - len(ts_list)

    # ts_matrix = np.array(ts_list)
    td_matrix = np.array(t_d)
    td_sort = td_matrix.argsort()
    # print(ts_matrix)

    for i in range(ts_lack):
        shift = 0
        ma_i = [0 for j in range(NFFT)]

        for j in range(i):
            if td_sort[i] > td_sort[j]:
                shift = shift + 1

        index = td_sort[i] + shift

        for j in range(NFFT):                       # 取索引位前后的中值作为插值
            ma_i[j] = (abs(matrix_list[index][j]) + abs(matrix_list[index + 1][j])) / 2

        matrix_list.insert(index + 1, ma_i)
        # ts_matrix = np.array(ts_list)

    matrix = np.array(matrix_list)
    ts_matrix = np.array(ts_list)
    f.close()

    # 写入npy文件
    with open("data/" + pcap_file + ".npy", 'wb'):
        pass
    with open("data_ts/" + pcap_file + ".npy", 'wb'):
        pass
    np.save("data/" + pcap_file + ".npy", matrix)
    np.save("data_ts/" + pcap_file + ".npy", ts_matrix)
    # 画图显示每秒包数
    # parse_draw(ts_matrix)

def parse_draw(ts_matrix):
    # ts_unique = np.unique(ts_matrix)
    # xnum = len(ts_unique)
    # no = [i for i in range(xnum + 1)]
    # x = [(str(no[i]) + '-' + str(no[i + 1])) for i in range(xnum)]
    xnum = ts_matrix[len(ts_matrix) - 1]
    x = [i + 1 for i in range(xnum + 1)]
    data_count = []
    data = sorted(ts_matrix)
    for i in range(xnum + 1):
        print(data.count(i))
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
    ax.set_title(pcap_name)
    for b in bn:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),b.get_height(), fontsize=7,ha = 'center',va='bottom')
    plt.show()

if __name__ == '__main__':
    # for i in range(1, 51):
    #     print("processing pcap: ", i)
    #     parse_pcap('static' + str(i))
    #     parse_pcap('circle' + str(i))
    #     parse_pcap('clap' + str(i))
    parse_pcap(pcap_name)
