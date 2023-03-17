import os
from model import *
from process import *
from pcap_proc import pcap_proc

fpath_NonVPN = './VPN/NonVPN-PCAPs-01/'
fpath_VPN = './VPN/VPN-PCAPs-01/'
fpath_VPN2 = './VPN/VPN-PCAPs-02/'
test_file = 'vpn_aim_chat1a.pcap'
fnum = 20
stream_num = 10000
FedRatio = 2
ratio = 5

def exFed(streams: dict):
    train_set, val_set, test_set = stream_to_data(streams)
    print(len(train_set))
    print(len(val_set))
    Cloud = myCloud(train_set, val_set, test_set)
    for i in range(FedRatio):
        estreams = dict([(key, streams[key]) for key in
                         np.random.choice(list(streams.keys()), int(len(streams) * 0.9), replace=False)])
        etrain_set, eval_set, _ = stream_to_data(estreams)
        Cloud.Edge.append(myEdge(etrain_set, eval_set, i))
        # print(len(Cloud.Edge[i].train_set))
        for j in range(FedRatio):
            cstreams = dict([(key, streams[key]) for key in
                             np.random.choice(list(estreams.keys()), int(len(estreams) * 0.8), replace=False)])
            ctrain_set, cval_set, _ = stream_to_data(cstreams)
            Cloud.Edge[i].Client.append(myClient(ctrain_set, cval_set, i * 2 + j))
            # print(len(Cloud.Edge[i].Client[j].train_set))
    Cloud_train(Cloud, 10, ratio)
    test(Cloud)

def rawFed(streams: dict):
    train_set, val_set, test_set = stream_to_data(streams)
    print(len(train_set))
    print(len(val_set))
    Server = rawFedServer(train_set, val_set, test_set)
    for i in range(FedRatio):
        for j in range(FedRatio):
            cstreams = dict([(key, streams[key]) for key in
                             np.random.choice(list(streams.keys()), int(len(streams) * 0.7), replace=False)])
            ctrain_set, cval_set, _ = stream_to_data(cstreams)
            Server.Client.append(rawFedClient(ctrain_set, cval_set))
    Fed_train(Server, 50, ratio)
    Fed_test(Server)

if __name__ == "__main__":
    mystreams = {}
    NonVPNfiles = os.listdir(fpath_NonVPN)
    VPNfiles = os.listdir(fpath_VPN)
    VPNfiles2 = os.listdir(fpath_VPN2)
    NonVPNfiles = NonVPNfiles[:max(len(NonVPNfiles), fnum)]
    VPNfiles = VPNfiles[:max(len(NonVPNfiles), fnum)]
    VPNfiles2 = VPNfiles2[:max(len(NonVPNfiles), fnum)]
    for file in NonVPNfiles:
        mystreams.update(pcap_proc(os.path.join(fpath_NonVPN, file), cnt=stream_num, isVPN=False))
    for file in VPNfiles:
        mystreams.update(pcap_proc(os.path.join(fpath_VPN, file), cnt=stream_num, isVPN=True))
    for file in VPNfiles2:
        mystreams.update(pcap_proc(os.path.join(fpath_VPN2, file), cnt=stream_num, isVPN=True))
    print(len(mystreams))
    rawFed(mystreams)

    # print(len(train_set), len(val_set), len(test_set))
    # train(myHDR, myPAY, mySR, myEnc, train_set, val_set)
    # local_train(myHDR, myPAY, mySR, myEnc, myCLoss, myELoss, ratio, train_set, val_set)
    # test(myHDR, myPAY, mySR, myEnc, test_set)

