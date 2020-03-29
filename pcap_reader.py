import scapy
from scapy.all import *
from scapy.utils import PcapReader
packets = rdpcap("./training_data/test.pcap")
for data in packets:
    if 'UDP' in data:
        s = repr(data)
        print(s)
        print(data['UDP'].sport)
        break
