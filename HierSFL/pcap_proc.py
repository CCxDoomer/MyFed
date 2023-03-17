import hashlib
from scapy.all import *

noLog = False
Nb = 784
Nbp = 784
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
    def __init__(self, src, sport, dst, dport, protol="TCP", isVPN=False):
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
        self.VPN = isVPN
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
        tpl_packet = packet.payload
        # tpl_packet = ip_packet.payload
        app_packet = tpl_packet.payload
        original_payload = app_packet.original
        # 负载长度
        packet_information.append(len(original_payload) / 65535.)
        # TCP窗口大小
        try:
            packet_information.append((tpl_packet.window if self.protol == "TCP" else 0) / 65535.)
        except:
            packet_information.append(0.)
        # print(tpl_packet.window if self.protol == "TCP" else 0)
        # IAT
        packet_information.append(math.log(1 + timestamp - self.next_time, math.e))
        self.next_time = timestamp
        # 数据包方向 0与流方向相同，1与流方向相反
        packet_information.append(0. if packet["IP"].src == self.src else 1.)
        # 负载部分
        # 总共取前Nb个字节的负载，每个包取前Nbp个字节的负载
        PAY = []
        if len(original_payload) > Nbp:
            original_payload = original_payload[:Nbp]
        if not self.isPayFull:
            if len(original_payload) + self.pay_len <= Nb:
                self.pay_len += len(original_payload)
            else:
                self.isPayFull = True
                original_payload = original_payload[:Nb - self.pay_len]
        else:
            original_payload = b''
        # 负载0-1标准化
        for item in original_payload:
            PAY.append(item / 255.)
        self.HDR.append(packet_information)
        self.PAY.append(PAY)


def read_pcap(streams, pcapname, cnt=5000, isVPN=False):
    try:
        packets = rdpcap(pcapname, cnt)
    except:
        print("error")
        return
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
        src, sport, dst, dport = data['IP'].src, data[protol].sport, \
                                 data['IP'].dst, data[protol].dport
        if protol == "UDP" and dst == '255.255.255.255' and dport == 10505:
            continue
        src, sport, dst, dport = NormalizationSrcDst(src, sport, dst, dport)
        stream_hash = tuple2hash(src, sport, dst, dport, protol)
        if stream_hash not in streams:
            streams[stream_hash] = Stream(src, sport, dst, dport, protol, isVPN)
        if streams[stream_hash].packet_num < Np:
            streams[stream_hash].add_packet(data)


def pcap_proc(filename='test.pcap', cnt=5000, isVPN=False):
    pcapname = filename
    streams = {}
    read_pcap(streams, pcapname, cnt, isVPN=isVPN)
    if len(streams) <= 1:
        return {}
    return streams
