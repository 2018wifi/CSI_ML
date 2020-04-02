import dpkt
import time
import struct
import numpy as np

BW = 20
NFFT = int(BW * 3.2)

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
    for ts, buf in pcap:
        eth = dpkt.ethernet.Ethernet(buf)
        ip = eth.data
        transf_data = ip.data
        payload = transf_data.data      # 逐层解包

        if len(payload) != 274:         # 大小不对时舍弃这个包
            continue

        data = parse_udp(payload)
        csi = read_csi(data)
        matrix_list.append(csi)
        ts_list.append(time.strftime("%H:%M:%S",time.localtime(ts)))
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

if __name__ == '__main__':
    for i in range(1, 51):
        print("processing pcap: ", i)
        parse_pcap('static' + str(i))
        parse_pcap('circle' + str(i))
        parse_pcap('clap' + str(i))