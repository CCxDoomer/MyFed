import torch
import Model
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pcap_proc import Stream, pcap_proc

device = "cuda" #if torch.cuda.is_available() else "cpu"

class myPreDataSet(object):
    def __init__(self, X):
        self.X = X
        self.mydata = X

    def __getitem__(self, idx):
        return self.mydata[idx]

    def __len__(self):
        return len(self.mydata)


def stream_to_data(streams: list):
    HDR = []
    PAY = []
    for stream_hash in streams:
        stream = streams[stream_hash]
        for hdr in stream.HDR:
            HDR.append(hdr)
        for pay in stream.PAY:
            PAY.append(pay)
    HDR = np.array(HDR, dtype=np.float32)
    PAY = np.array(PAY, dtype=np.float32)
    tHDR = HDR.copy()
    tPAY = PAY.copy()
    np.random.shuffle(HDR)
    np.random.shuffle(PAY)
    HDR_div1 = int(len(HDR) * 0.6)
    HDR_div2 = int(len(HDR) * 0.8)
    PAY_div1 = int(len(PAY) * 0.6)
    PAY_div2 = int(len(PAY) * 0.8)
    HDR_train_set = myPreDataSet(HDR[:HDR_div1])
    HDR_val_set = myPreDataSet(HDR[HDR_div1:HDR_div2])
    HDR_test_set = myPreDataSet(HDR[HDR_div2:])
    PAY_train_set = myPreDataSet(PAY[:PAY_div1])
    PAY_val_set = myPreDataSet(PAY[PAY_div1:PAY_div2])
    PAY_test_set = myPreDataSet(PAY[PAY_div2:])
    return tHDR, tPAY, [HDR_train_set, HDR_val_set, HDR_test_set], \
                       [PAY_train_set, PAY_val_set, PAY_test_set]


def pretrain_HAE(HDR_set: list, myHAE: Model.HAE):
    myHAE.train()
    train_loader = DataLoader(HDR_set[0], batch_size=32, shuffle=False)
    val_loader = DataLoader(HDR_set[1], batch_size=32, shuffle=False)
    optimizer = torch.optim.Adam(myHAE.parameters(), lr=1e-3, )
    loss_fn = nn.MSELoss()
    for epoch in tqdm(range(200)):
        train_loss = []
        for x in train_loader:
            x = x.to(device)
            x_ = myHAE(x)
            loss = loss_fn(x_, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())
        train_loss = np.mean(train_loss)
        myHAE.eval()
        val_loss = []
        for x in val_loader:
            x = x.to(device)
            x_ = myHAE(x)
            x = x.cpu(); x_ = x_.cpu()
            loss = loss_fn(x_, x)
            val_loss.append(loss.cpu().item())
        val_loss = np.mean(val_loss)
        myHAE.train()
        tqdm.write('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, train_loss, val_loss))

streams = pcap_proc()
HDR, PAY, HDR_set, PAY_et = stream_to_data(streams)
myHAE = Model.HAE().to(device)
pretrain_HAE(HDR_set, myHAE)
