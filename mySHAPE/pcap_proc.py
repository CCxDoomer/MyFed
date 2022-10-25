import argparse
import hashlib
from scapy.all import *

noLog = False
Nb = 784
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
        # packet_head = ""
        # if packet["IP"].src == self.src:
        #     # 代表这是一个客户端发往服务器的包
        #     packet_head += "---> "
        #     if self.protol == "TCP":
        #         # 对TCP包额外处理
        #         packet_head += "[{:^4}] ".format(str(packet['TCP'].flags))
        #         if self.packet_num == 1 or packet['TCP'].flags == "S":
        #             # 对一个此流的包或者带有Syn标识的包的时间戳进行记录，作为starttime
        #             self.start_time = timestamp
        # else:
        #     packet_head += "<--- "
        # packet_information = packet_head + "timestamp={}".format(timestamp)
        packet_information = list()
        ip_packet = packet.payload
        tpl_packet = ip_packet.payload
        app_packet = tpl_packet.payload
        original_payload = app_packet.original
        # 负载长度
        packet_information.append(len(original_payload))
        # TCP窗口大小
        packet_information.append(tpl_packet.window if self.protol == "TCP" else 0)
        # IAT
        packet_information.append(math.log(1 + timestamp - self.next_time, math.e))
        self.next_time = timestamp
        # 数据包方向 0与流方向相同，1与流方向相反
        packet_information.append(0 if packet["IP"].src == self.src else 0)
        # 负载部分
        PAY = []
        if len(original_payload) > 128:
            original_payload = original_payload[:128]
        if not self.isPayFull:
            if len(original_payload) + self.pay_len <= Nb:
                self.pay_len += len(original_payload)
                self.LPi.append(len(original_payload))
            else:
                self.isPayFull = True
                original_payload = original_payload[:Nb-self.pay_len]
                self.LPi.append(Nb-self.pay_len)
        else:
            original_payload = b''
            self.LPi.append(0.)
        for item in original_payload:
            PAY.append(item/255.)
        self.HDR.append(packet_information)
        self.PAY.append(PAY)

    def get_timestamp(self, packet):
        if packet['IP'].proto == 'udp':
            # udp协议查不到时间戳
            return 0
        for t in packet['TCP'].options:
            if t[0] == 'Timestamp':
                return t[1][0]
        # 存在查不到时间戳的情况
        return -1

    def __repr__(self):
        return "{} {}:{} -> {}:{} {} {} {}".format(self.protol, self.src,
                                                   self.sport, self.dst,
                                                   self.dport, self.byte_num,
                                                   self.start_time, self.end_time)


# pcapname：输入pcap的文件名
# csvname : 输出csv的文件名
def read_pcap(pcapname):
    try:
        # 可能存在格式错误读取失败的情况
        packets = rdpcap(pcapname)
    except:
        print("read pcap error")
        return
    global streams
    streams = {}
    for data in packets:
        try:
            # 抛掉不是IP协议的数据包
            data['IP']
        except:
            continue
        if data['IP'].proto == 6:
            protol = "TCP"
        elif data['IP'].proto == 17:
            protol = "UDP"
        else:
            # 非这两种协议的包，忽视掉
            continue
        src, sport, dst, dport = NormalizationSrcDst(data['IP'].src, data[protol].sport,
                                                     data['IP'].dst, data[protol].dport)
        hash_str = tuple2hash(src, sport, dst, dport, protol)
        if hash_str not in streams:
            streams[hash_str] = Stream(src, sport, dst, dport, protol)
        if streams[hash_str].packet_num < Np:
            streams[hash_str].add_packet(data)
    for stream_hash in streams:
        stream = streams[stream_hash]
        max_pay_len = max(stream.LPi)
        max_pay_len = max_pay_len + 8 - (max_pay_len % 8)
        for i in range(len(stream.LPi)):
            pay_len = len(stream.PAY[i])
            pad_len = max_pay_len - pay_len
            stream.PAY[i].extend([0 for j in range(pad_len)])
    # with open(csvname, "a+", newline="") as file:
    #     writer = csv.writer(file)
    #     for v in streams.values():
    #         writer.writerow((
    #             time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(v.start_time)), v.end_time - v.start_time,
    #             v.src, v.sport, v.dst, v.dport,
    #             v.packet_num, v.byte_num, v.byte_num / v.packet_num, v.protol))
    #         if noLog == False:
    #             print(v)


def pcap_proc():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pcap", help="pcap文件名", action='store', default='test.pcap')
    parser.add_argument("-o", "--output", help="输出的csv文件名", action='store', default="stream.csv")
    parser.add_argument("-n", "--nolog", action='store_true', help='读取当前文件夹下的所有pcap文件', default=False)
    parser.add_argument("-t", "--test", action='store_true', default=False)
    args = parser.parse_args()
    csvname = args.output
    noLog = args.nolog
    pcapname = args.pcap
    read_pcap(pcapname)
    global streams
    return streams

