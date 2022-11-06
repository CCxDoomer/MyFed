import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
from Model import HAE, PAE, MultAttention, N_emb_dims, N_hidden
from torch.utils.data import DataLoader
from pcap_proc import Stream, pcap_proc

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32
E_HAE = 10
E_PAE = 10
N_Pmax = 64
N_layers = 4
jBATCH = 4


# 自定义dataset
class myPreDataSet(object):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y
        if Y is None:
            self.mydata = X
        else:
            self.mydata = [(x, y) for x, y in zip(X, Y)]

    def __getitem__(self, idx):
        return self.mydata[idx]

    def __len__(self):
        return len(self.mydata)


# 将所有HDR和PAE合并并划分数据集
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
    # shuffle
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


# 对token划分流完整性数据集
def token_to_preint_data(np_tokens: list):
    tokens = []
    for np_token in np_tokens:
        tokens.append(np_token)
    tokens = np.array(tokens)
    # 0表示未发生替换、1表示发生替换
    replace_flag = np.array([0 for j in range(len(tokens))], dtype=np.float32)
    # 0.5的概率将流中的包替换为另一个流的数据
    for i in range(0, len(tokens)):
        replace_rand = np.random.rand()
        if replace_rand <= 0.5:
            replace_flag[i] = 0
        else:
            replace_flag[i] = 1
            # 流替换算法
    index = list(range(len(tokens)))
    # shuffle
    np.random.shuffle(index)
    tokens = tokens[index]
    replace_flag = replace_flag[index]
    token_div1 = int(len(tokens) * 0.6)
    token_div2 = int(len(tokens) * 0.8)
    tokens_train_set = myPreDataSet(tokens[:token_div1], replace_flag[:token_div1])
    tokens_val_set = myPreDataSet(tokens[token_div1:token_div2], replace_flag[token_div1:token_div2])
    tokens_test_set = myPreDataSet(tokens[token_div2:], replace_flag[token_div2:])
    return [tokens_train_set, tokens_val_set, tokens_test_set]


# 对token划分流重建数据集
def token_to_prerec_data(np_tokens: list):
    tokens = []
    # 0.07概率将一列变为(0,1)随机数、0.28概率将一列变为0
    for np_token in np_tokens:
        change_rand = np.random.rand()
        pos_rand = np.random.randint(1, np_token.shape[1])
        if change_rand <= 0.07:
            np_token[:, pos_rand] = np.random.rand(np_token.shape[0])
        if change_rand >= 1 - 0.28:
            np_token[:, pos_rand] = np.zeros(np_token.shape[0])
        tokens.append(np_token)
    tokens = np.array(tokens)
    token_div1 = int(len(tokens) * 0.6)
    token_div2 = int(len(tokens) * 0.8)
    tokens_train_set = myPreDataSet(tokens[:token_div1])
    tokens_val_set = myPreDataSet(tokens[token_div1:token_div2])
    tokens_test_set = myPreDataSet(tokens[token_div2:])
    return [tokens_train_set, tokens_val_set, tokens_test_set]


# 预训练HAE
def pretrain_HAE(HDR_set: list, myHAE: HAE):
    print(f"Pretraining HAE")
    myHAE.train()
    train_loader = DataLoader(HDR_set[0], batch_size=BATCH, shuffle=False)
    val_loader = DataLoader(HDR_set[1], batch_size=BATCH, shuffle=False)
    optimizer = torch.optim.Adam(myHAE.parameters(), lr=1e-3, )
    loss_fn = nn.MSELoss()
    for epoch in tqdm(range(E_HAE)):
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
            x = x.cpu()
            x_ = x_.cpu()
            loss = loss_fn(x_, x)
            val_loss.append(loss.cpu().item())
        val_loss = np.mean(val_loss)
        myHAE.train()
        tqdm.write('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, train_loss, val_loss))


# 预训练PAE
def pretrain_PAE(PAE_set: list, myPAE: PAE):
    print(f"Pretraining PAE")
    myPAE.train()
    train_loader = DataLoader(PAE_set[0], batch_size=BATCH, shuffle=False)
    val_loader = DataLoader(PAE_set[1], batch_size=BATCH, shuffle=False)
    optimizer = torch.optim.Adam(myPAE.parameters(), lr=1e-4, )
    loss_fn = nn.MSELoss()
    for epoch in tqdm(range(E_PAE)):
        train_loss = []
        for x in train_loader:
            x = x.to(device)
            x = x.unsqueeze(1)
            x_ = myPAE(x)
            x = x.squeeze()
            loss = loss_fn(x_, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())
        train_loss = np.mean(train_loss)
        myPAE.eval()
        val_loss = []
        for x in val_loader:
            x = x.to(device)
            x = x.unsqueeze(1)
            x_ = myPAE(x)
            x = x.squeeze()
            x = x.cpu()
            x_ = x_.cpu()
            loss = loss_fn(x_, x)
            val_loss.append(loss.cpu().item())
        val_loss = np.mean(val_loss)
        myPAE.train()
        tqdm.write('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, train_loss, val_loss))


# 预训练，返回训练好的HAE和PAE
def pretrain(streams: list):
    HDR, PAY, HDR_set, PAY_set = stream_to_data(streams)
    myHAE = HAE().to(device)
    pretrain_HAE(HDR_set, myHAE)
    torch.save(myHAE, 'myHAE.pth')
    myPAE = PAE().to(device)
    pretrain_PAE(PAY_set, myPAE)
    torch.save(myPAE, 'myPAE.pth')
    return myHAE, myPAE


def generate_mask():
    pe_mask = list()
    for i in range(N_emb_dims):
        pe_mask.append([])
        for j in range(N_Pmax):
            if i % 2 == 0:
                pe = math.sin(j / pow(10000., i // 2 / N_hidden))
            else:
                pe = math.cos(j / pow(10000., i // 2 / N_hidden))
            pe_mask[i].append(pe)
    pe_mask = torch.FloatTensor(pe_mask).cpu()
    return pe_mask


# 将流转变为token
def strean_to_token(streams: list, myHAE: HAE, myPAE: PAE, mask: torch.Tensor):
    tokens = []
    myHAE.eval()
    myPAE.eval()
    cnt = 0
    for stream_hash in streams:
        stream: Stream = streams[stream_hash]
        token = torch.tensor([]).cpu()
        token_hat = []
        HDR = stream.HDR
        PAY = stream.PAY
        LPi = stream.LPi
        # Encode
        for i in range(len(PAY)):
            # print(i)
            token_hat.append([1, 0])
            encHDR = myHAE.Encode(torch.tensor(HDR[i]).unsqueeze(0).to(device))
            encHDR = encHDR.T.cpu()
            if LPi[i] == 0:
                token = torch.cat([token, encHDR], 1)
                continue
            encLPi = math.ceil(LPi[i] / 8.)
            for j in range(encLPi):
                token_hat.append([0, 1])
            encPAY = myPAE.Encode(torch.tensor(PAY[i]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(1).to(device))
            encPAY = encPAY.squeeze().cpu()
            encPAY = encPAY[:, :encLPi]
            token = torch.cat([token, encHDR, encPAY], 1)
        for i in range(len(PAY), len(HDR)):
            encHDR = myHAE.Encode(torch.tensor(HDR[i]).unsqueeze(0).to(device))
            encHDR = encHDR.cpu()
            token_hat.append([0, 1])
            token = torch.cat([token, encHDR], 1)
        # Embed
        token_hat = torch.from_numpy(np.array(token_hat))
        token_hat = token_hat.transpose(1, 0)
        token = torch.cat([token_hat, token], 0)
        CLS = torch.zeros(N_emb_dims, 1)
        token = torch.cat([CLS, token], 1)
        row, col = token.size()
        if col < N_Pmax:
            token = torch.cat([token, torch.zeros(row, N_Pmax - col)], 1)
        else:
            token = token[:, :N_Pmax]
        # print(row, col)
        # print(token.size())
        token = token + mask
        np.savetxt(f'./tokens/{cnt}.csv', token.detach().numpy(), delimiter=',')
        cnt += 1
        tokens.append(token)
    return tokens


# 对transformer进行联合训练
def joint_pretrain(np_tokens: list, myMult: MultAttention, myHAE: HAE, myPAE: PAE):
    tokens_integrity_set = token_to_preint_data(np_tokens)
    int_train_loader = DataLoader(tokens_integrity_set[0], batch_size=jBATCH, shuffle=False)
    int_val_loader = DataLoader(tokens_integrity_set[1], batch_size=jBATCH, shuffle=False)
    tokens_reconstruct_set = token_to_prerec_data(np_tokens)
    rec_train_loader = DataLoader(tokens_reconstruct_set[0], batch_size=jBATCH, shuffle=False)
    rec_val_loader = DataLoader(tokens_reconstruct_set[1], batch_size=jBATCH, shuffle=False)


if __name__ == '__main__':
    streams = pcap_proc()
    print(len(streams))
    # myHAE, myPAE = pretrain(streams)
    myHAE = torch.load('myHAE.pth').to(device)
    myPAE = torch.load('myPAE.pth').to(device)
    # tokens = strean_to_token(streams, myHAE, myPAE, generate_mask())
    np_tokens = []
    for i in range(len(streams)):
        np_tokens.append(np.loadtxt(f'./tokens/{i}.csv', dtype=np.float32, delimiter=','))
    myMultAttn = MultAttention(N_layers, N_Pmax).to(device)
    joint_pretrain(np_tokens, myMultAttn, myHAE, myPAE)
