import argparse
import hashlib
from scapy.all import *

noLog = False
Nb = 784
Nbp = 128
Np = 32


# 根据规则区分服务器和客户端
def NormalizationSrcDst(src, sport, dst, dport):
    if sport < dport:
        return dst, dport, src, sport
    elif sport == dport:
        src_ip = "".join(src.split('.'))
        dst_ip = "".join(dst.split('.'))
        if int(src_ip) < int(dst_ip):
            return dst, dport, src, sport
        else:
            return src, sport, dst, dport
    else:
        return src, sport, dst, dport


# 将五元组信息转换为MD5值,用于字典存储
def tuple2hash(src, sport, dst, dport, protocol):
    hash_str = src + str(sport) + dst + str(dport) + protocol
    return hashlib.md5(hash_str.encode(encoding="UTF-8")).hexdigest()


class Stream:
    def __init__(self, src, sport, dst, dport, protol="TCP"):
        self.src = src
        self.sport = sport
        self.dst = dst
        self.dport = dport
        self.protol = protol
        self.start_time = 0
        self.next_time = 0
        self.end_time = 0
        self.packet_num = 0
        self.byte_num = 0
        self.HDR = []
        self.PAY = []
        self.LPi = []
        self.pay_len = 0
        self.isPayFull = False

    def add_packet(self, packet):
        # 在当前流下新增一个数据包
        self.packet_num += 1
        self.byte_num += len(packet)
        timestamp = packet.time  # 浮点型
        if self.start_time == 0:
            # 如果starttime还是默认值0，则立即等于当前时间戳
            self.start_time = timestamp
        if self.next_time == 0:
            self.next_time = timestamp
        self.start_time = min(timestamp, self.start_time)
        self.end_time = max(timestamp, self.end_time)
        packet_information = list()
        ip_packet = packet.payload
        tpl_packet = ip_packet.payload
        app_packet = tpl_packet.payload
        original_payload = app_packet.original
        # 负载长度
        packet_information.append(len(original_payload) / 65535.)
        # TCP窗口大小
        packet_information.append((tpl_packet.window if self.protol == "TCP" else 0) / 65535.)
        # print(tpl_packet.window if self.protol == "TCP" else 0)
        # IAT
        packet_information.append(math.log(1 + timestamp - self.next_time, math.e))
        self.next_time = timestamp
        # 数据包方向 0与流方向相同，1与流方向相反
        packet_information.append(0 if packet["IP"].src == self.src else 1)
        # 负载部分
        # 总共取前Nb个字节的负载，每个包取前Nbp个字节的负载
        PAY = []
        if len(original_payload) > Nbp:
            original_payload = original_payload[:Nbp]
        if not self.isPayFull:
            if len(original_payload) + self.pay_len <= Nb:
                self.pay_len += len(original_payload)
                self.LPi.append(len(original_payload))
            else:
                self.isPayFull = True
                original_payload = original_payload[:Nb - self.pay_len]
                self.LPi.append(Nb - self.pay_len)
        else:
            original_payload = b''
            self.LPi.append(0.0)
        # 负载0-1标准化
        for item in original_payload:
            PAY.append(item / 255.)
        self.HDR.append(packet_information)
        self.PAY.append(PAY)


def read_pcap(pcapname):
    try:
        packets = rdpcap(pcapname)
    except:
        print("error")
        return
    global streams
    streams = {}
    for data in packets:
        try:
            data['IP']
        except:
            continue
        if data['IP'].proto == 6:
            protol = "TCP"
        elif data['IP'].proto == 17:
            protol = "UDP"
        else:
            continue
        src, sport, dst, dport = NormalizationSrcDst(data['IP'].src, data[protol].sport,
                                                     data['IP'].dst, data[protol].dport)
        stream_hash = tuple2hash(src, sport, dst, dport, protol)
        if stream_hash not in streams:
            streams[stream_hash] = Stream(src, sport, dst, dport, protol)
        if streams[stream_hash].packet_num < Np:
            streams[stream_hash].add_packet(data)
    # PAY填充至最长PAYi的长度，同时满足长度整除8
    for stream_hash in streams:
        stream = streams[stream_hash]
        max_pay_len = Nbp  # max(stream.LPi)
        # max_pay_len = max_pay_len + 8 - (max_pay_len % 8)
        for i in range(len(stream.LPi)):
            pay_len = len(stream.PAY[i])
            pad_len = max_pay_len - pay_len
            stream.PAY[i].extend([0 for j in range(pad_len)])


def pcap_proc(filename='test.pcap', ):
    pcapname = filename
    read_pcap(pcapname)
    global streams
    return streams
