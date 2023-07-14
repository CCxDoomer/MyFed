import torch
import torch.nn as nn
import math
import time
import numpy as np
from pcap_proc import Np, Nb
from model import *
from sklearn.metrics import f1_score, confusion_matrix

BATCH = 16

def stream_to_data(streams: dict):
    HDRs = []
    PAYs = []
    tokens = []
    for stream_hash in streams:
        stream = streams[stream_hash]
        HDR = []
        PAY = []
        for hdr in stream.HDR:
            HDR.extend(hdr)
        for pay in stream.PAY:
            PAY.extend(pay)
        for i in range(len(HDR), Np * 4):
            HDR.append(0.)
        for i in range(len(PAY), Nb):
            PAY.append(0.)
        HDR = np.array(HDR, dtype=np.float32)
        PAY = np.array(PAY, dtype=np.float32)
        HDRs.append(HDR)
        PAYs.append(PAY)
        tokens.append(1 if stream.VPN else 0)
    # shuffle
    # print(sum(tokens))
    HDRs = np.array(HDRs, dtype=np.float32)
    PAYs = np.array(PAYs, dtype=np.float32)
    tokens = np.array(tokens, dtype=np.float32)
    index = list(range(len(HDRs)))
    np.random.shuffle(index)
    HDRs = HDRs[index]
    PAYs = PAYs[index]
    tokens = tokens[index]
    div1 = int(len(HDRs) * 0.6)
    div2 = int(len(HDRs) * 0.8)
    train_set = myPreDataSet(HDRs[:div1], PAYs[:div1], tokens[:div1])
    val_set = myPreDataSet(HDRs[div1:div2], PAYs[div1:div2], tokens[div1:div2])
    test_set = myPreDataSet(HDRs[div2:], PAYs[div2:], tokens[div2:])
    return train_set, val_set, test_set

def Cloud_train(Cloud: myCloud, Epoch: int, ratio: int):
    init_time = time.time()
    loss_fn = nn.CrossEntropyLoss()
    for e in range(Epoch):
        for Edge in Cloud.Edge:
            Edge_train(Edge, ratio)
        for cParam in Cloud.HDR.parameters():
            cParam.data.zero_()
        for cParam in Cloud.PAY.parameters():
            cParam.data.zero_()
        for cParam in Cloud.SR.parameters():
            cParam.data.zero_()
        for Edge in Cloud.Edge:
            for cParam, eParam in zip(Cloud.HDR.parameters(), Edge.HDR.parameters()):
                cParam.data += eParam.data / len(Cloud.Edge)
            for cParam, eParam in zip(Cloud.PAY.parameters(), Edge.PAY.parameters()):
                cParam.data += eParam.data / len(Cloud.Edge)
            for cParam, eParam in zip(Cloud.SR.parameters(), Edge.SR.parameters()):
                cParam.data += eParam.data / len(Cloud.Edge)
        for Edge in Cloud.Edge:
            for cParam, eParam in zip(Cloud.HDR.parameters(), Edge.HDR.parameters()):
                eParam.data = cParam.data
            for cParam, eParam in zip(Cloud.PAY.parameters(), Edge.PAY.parameters()):
                eParam.data = cParam.data
            for cParam, eParam in zip(Cloud.SR.parameters(), Edge.SR.parameters()):
                eParam.data = cParam.data
        train_loss = []; train_acc = []; val_loss = []; val_acc = []
        f1 = []
        Cloud.HDR.train(); Cloud.PAY.train(); Cloud.SR.train(); Cloud.Enc.train()
        for hdr, pay, y in Cloud.train_set:
            y = y.type(torch.LongTensor)
            hdr = hdr.to(device)
            pay = pay.to(device)
            y = y.to(device)
            tmp1 = Cloud.HDR(hdr)
            tmp2 = Cloud.PAY(pay)
            if tmp1.shape[0] == 1:
                continue
            tmp = Cloud.SR(tmp2, tmp1)
            y_ = Cloud.Enc(tmp)
            loss = loss_fn(y_, y)
            Cloud.optimizer_HDR.zero_grad(); Cloud.optimizer_PAY.zero_grad()
            Cloud.optimizer_SR.zero_grad(); Cloud.optimizer_Enc.zero_grad()
            loss.backward()
            Cloud.optimizer_HDR.step(); Cloud.optimizer_PAY.step()
            Cloud.optimizer_SR.step(); Cloud.optimizer_Enc.step()
            y_ = torch.max(y_, -1)[1]
            train_acc.append(y_.eq(y.data.view_as(y_)).long().cpu().sum() / y_.shape[0])
            train_loss.append(loss.cpu().item())
        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
        Cloud.HDR.eval(); Cloud.PAY.eval(); Cloud.SR.eval(); Cloud.Enc.eval()
        for hdr, pay, y in Cloud.val_set:
            y = y.type(torch.LongTensor)
            hdr = hdr.to(device)
            pay = pay.to(device)
            y = y.to(device)
            tmp1 = Cloud.HDR(hdr)
            tmp2 = Cloud.PAY(pay)
            if tmp1.shape[0] == 1:
                continue
            tmp = Cloud.SR(tmp2, tmp1)
            y_ = Cloud.Enc(tmp)
            loss = loss_fn(y_, y)
            y_ = torch.max(y_, -1)[1]
            val_acc.append(y_.eq(y.data.view_as(y_)).long().cpu().sum() / y_.shape[0])
            val_loss.append(loss.cpu().item())
            f1.append(f1_score(y.cpu(), y_.cpu(), ))
        val_loss = np.mean(val_loss); val_acc = np.mean(val_acc)
        f1 = np.mean(f1)
        print(f"Cloud_{e}: {train_loss} {train_acc} {val_loss} {val_acc} {time.time()} {f1}")

def Edge_train(Edge: myEdge, ratio: int):
    init_time = time.time()
    loss_fn = nn.CrossEntropyLoss()
    for e in range(ratio):
        for Client in Edge.Client:
            Client_train(Client, ratio)
        for eParam in Edge.HDR.parameters():
            eParam.data.zero_()
        for eParam in Edge.PAY.parameters():
            eParam.data.zero_()
        for Client in Edge.Client:
            for eParam, cParam in zip(Edge.HDR.parameters(), Client.HDR.parameters()):
                eParam.data += cParam.data / len(Edge.Client)
            for eParam, cParam in zip(Edge.PAY.parameters(), Client.PAY.parameters()):
                eParam.data += cParam.data / len(Edge.Client)
        for Client in Edge.Client:
            for eParam, cParam in zip(Edge.HDR.parameters(), Client.HDR.parameters()):
                cParam.data = eParam.data
            for eParam, cParam in zip(Edge.PAY.parameters(), Client.PAY.parameters()):
                cParam.data = eParam.data
        train_loss = []; val_loss = []
        Edge.HDR.train(); Edge.PAY.train(); Edge.SR.train(); Edge.ELoss.train()
        for hdr, pay, y in Edge.train_set:
            y = y.type(torch.LongTensor)
            hdr = hdr.to(device)
            pay = pay.to(device)
            y = y.to(device)
            tmp1 = Edge.HDR(hdr)
            tmp2 = Edge.PAY(pay)
            if tmp1.shape[0] == 1:
                continue
            tmp = Edge.SR(tmp2, tmp1)
            y_ = Edge.ELoss(tmp)
            loss = loss_fn(y_, y)
            Edge.optimizer_HDR.zero_grad(); Edge.optimizer_PAY.zero_grad()
            Edge.optimizer_SR.zero_grad(); Edge.optimizer_Loss.zero_grad()
            loss.backward()
            Edge.optimizer_HDR.step(); Edge.optimizer_PAY.step()
            Edge.optimizer_SR.step(); Edge.optimizer_Loss.step()
            train_loss.append(loss.cpu().item())
        train_loss = np.mean(train_loss)
        Edge.HDR.eval(); Edge.PAY.eval(); Edge.SR.eval(); Edge.ELoss.eval()
        for hdr, pay, y in Edge.val_set:
            y = y.type(torch.LongTensor)
            hdr = hdr.to(device)
            pay = pay.to(device)
            y = y.to(device)
            tmp1 = Edge.HDR(hdr)
            tmp2 = Edge.PAY(pay)
            if tmp1.shape[0] == 1:
                continue
            tmp = Edge.SR(tmp2, tmp1)
            y_ = Edge.ELoss(tmp)
            loss = loss_fn(y_, y)
            val_loss.append(loss.cpu().item())
        val_loss = np.mean(val_loss)
        # print(f"Edge_{Edge.name}_{e}: {train_loss} {val_loss} {time.time()}")

def Client_train(Client: myClient, Epoch: int):
    init_time = time.time()
    loss_fn = nn.CrossEntropyLoss()
    for e in range(Epoch):
        train_loss = []; val_loss = []
        Client.HDR.train(); Client.PAY.train(); Client.CLoss.train()
        for hdr, pay, y in Client.train_set:
            y = y.type(torch.LongTensor)
            hdr = hdr.to(device)
            pay = pay.to(device)
            y = y.to(device)
            tmp1 = Client.HDR(hdr)
            tmp2 = Client.PAY(pay)
            if tmp1.shape[0] == 1:
                continue
            y_ = Client.CLoss(tmp2, tmp1)
            loss = loss_fn(y_, y)
            Client.optimizer_HDR.zero_grad(); Client.optimizer_PAY.zero_grad()
            Client.optimizer_Loss.zero_grad()
            loss.backward()
            Client.optimizer_HDR.step(); Client.optimizer_PAY.step()
            Client.optimizer_Loss.step()
            y_ = torch.max(y_, -1)[1]
            train_loss.append(loss.cpu().item())
        train_loss = np.mean(train_loss)
        Client.HDR.eval(); Client.PAY.eval(); Client.CLoss.eval()
        for hdr, pay, y in Client.val_set:
            y = y.type(torch.LongTensor)
            hdr = hdr.to(device)
            pay = pay.to(device)
            y = y.to(device)
            tmp1 = Client.HDR(hdr)
            tmp2 = Client.PAY(pay)
            if tmp1.shape[0] == 1:
                continue
            y_ = Client.CLoss(tmp2, tmp1)
            loss = loss_fn(y_, y)
            val_loss.append(loss.cpu().item())
        val_loss = np.mean(val_loss)
        # print(f"Client_{Client.name}_{e}: {train_loss} {val_loss} {time.time()}")

def test(Cloud: myCloud):
    test_loss = []; test_acc = []
    confussion_m = np.array([[0, 0], [0, 0]])
    loss_fn = nn.CrossEntropyLoss()
    Cloud.HDR.eval(); Cloud.PAY.eval(); Cloud.SR.eval(); Cloud.Enc.eval()
    for hdr, pay, y in Cloud.test_set:
        y = y.type(torch.LongTensor)
        hdr = hdr.to(device)
        pay = pay.to(device)
        y = y.to(device)
        tmp1 = Cloud.HDR(hdr)
        tmp2 = Cloud.PAY(pay)
        if tmp1.shape[0] == 1:
            continue
        tmp = Cloud.SR(tmp2, tmp1)
        y_ = Cloud.Enc(tmp)
        loss = loss_fn(y_, y)
        y_ = torch.max(y_, -1)[1]
        test_acc.append(y_.eq(y.data.view_as(y_)).long().cpu().sum() / y_.shape[0])
        test_loss.append(loss.cpu().item())
        confussion_m = np.add(confussion_m, confusion_matrix(y.cpu(), y_.cpu()))
    test_loss = np.mean(test_loss); test_acc = np.mean(test_acc)
    print(f"test: {test_loss} {test_acc}")
    print(f"confusion: {confussion_m}")

def Fed_train(Server: rawFedServer, Epoch: int, ratio: int):
    loss_fn = nn.CrossEntropyLoss()
    for e in range(Epoch):
        for i in range(len(Server.Client)):
            Client = Server.Client[i]
            for j in range(ratio):
                train_loss = []; val_loss = []
                Client.HDR.train(); Client.PAY.train(); Client.SR.train(); Client.Enc.train()
                for hdr, pay, y in Client.train_set:
                    y = y.type(torch.LongTensor)
                    hdr = hdr.to(device)
                    pay = pay.to(device)
                    y = y.to(device)
                    tmp1 = Client.HDR(hdr)
                    tmp2 = Client.PAY(pay)
                    tmp = torch.cat((tmp1, tmp2), 1)
                    if tmp.shape[0] == 1:
                        continue
                    tmp = Client.SR(tmp)
                    y_ = Client.Enc(tmp)
                    loss = loss_fn(y_, y)
                    Client.optimizer_HDR.zero_grad(); Client.optimizer_PAY.zero_grad()
                    Client.optimizer_SR.zero_grad(); Client.optimizer_Enc.zero_grad()
                    loss.backward()
                    Client.optimizer_HDR.step(); Client.optimizer_PAY.step()
                    Client.optimizer_SR.step(); Client.optimizer_Enc.step()
                    y_ = torch.max(y_, -1)[1]
                    train_loss.append(loss.cpu().item())
                train_loss = np.mean(train_loss)
                Client.HDR.eval(); Client.PAY.eval()
                Client.SR.eval(); Client.Enc.eval()
                for hdr, pay, y in Client.val_set:
                    y = y.type(torch.LongTensor)
                    hdr = hdr.to(device)
                    pay = pay.to(device)
                    y = y.to(device)
                    tmp1 = Client.HDR(hdr)
                    tmp2 = Client.PAY(pay)
                    tmp = torch.cat((tmp1, tmp2), 1)
                    if tmp.shape[0] == 1:
                        continue
                    tmp = Client.SR(tmp)
                    y_ = Client.Enc(tmp)
                    loss = loss_fn(y_, y)
                    val_loss.append(loss.cpu().item())
                val_loss = np.mean(val_loss)
                # print(f"Client_{i}_{j}: {train_loss} {val_loss} {time.time()}")
        for cParam in Server.HDR.parameters():
            cParam.data.zero_()
        for cParam in Server.PAY.parameters():
            cParam.data.zero_()
        for cParam in Server.SR.parameters():
            cParam.data.zero_()
        for cParam in Server.Enc.parameters():
            cParam.data.zero_()
        for Client in Server.Client:
            for cParam, eParam in zip(Server.HDR.parameters(), Client.HDR.parameters()):
                cParam.data += eParam.data / len(Server.Client)
            for cParam, eParam in zip(Server.PAY.parameters(), Client.PAY.parameters()):
                cParam.data += eParam.data / len(Server.Client)
            for cParam, eParam in zip(Server.SR.parameters(), Client.SR.parameters()):
                cParam.data += eParam.data / len(Server.Client)
            for cParam, eParam in zip(Server.Enc.parameters(), Client.Enc.parameters()):
                cParam.data += eParam.data / len(Server.Client)
        # for Client in Server.Client:
        #     for cParam, eParam in zip(Server.HDR.parameters(), Client.HDR.parameters()):
        #         eParam.data = cParam.data
        #     for cParam, eParam in zip(Server.PAY.parameters(), Client.PAY.parameters()):
        #         eParam.data = cParam.data
        #     for cParam, eParam in zip(Server.SR.parameters(), Client.SR.parameters()):
        #         eParam.data = cParam.data
        #     for cParam, eParam in zip(Server.Enc.parameters(), Client.Enc.parameters()):
        #         eParam.data = cParam.data
        train_loss = []; train_acc = []; val_loss = []; val_acc = []
        f1 = []
        Server.HDR.train(); Server.PAY.train(); Server.SR.train(); Server.Enc.train()
        for hdr, pay, y in Server.train_set:
            y = y.type(torch.LongTensor)
            hdr = hdr.to(device)
            pay = pay.to(device)
            y = y.to(device)
            tmp1 = Server.HDR(hdr)
            tmp2 = Server.PAY(pay)
            tmp = torch.cat((tmp1, tmp2), 1)
            if tmp.shape[0] == 1:
                continue
            tmp = Server.SR(tmp)
            y_ = Server.Enc(tmp)
            loss = loss_fn(y_, y)
            Server.optimizer_HDR.zero_grad(); Server.optimizer_PAY.zero_grad()
            Server.optimizer_SR.zero_grad(); Server.optimizer_Enc.zero_grad()
            loss.backward()
            Server.optimizer_HDR.step(); Server.optimizer_PAY.step()
            Server.optimizer_SR.step(); Server.optimizer_Enc.step()
            y_ = torch.max(y_, -1)[1]
            train_acc.append(y_.eq(y.data.view_as(y_)).long().cpu().sum() / y_.shape[0])
            train_loss.append(loss.cpu().item())
        train_loss = 0; train_acc = 0
        Server.HDR.eval(); Server.PAY.eval(); Server.SR.eval(); Server.Enc.eval()
        for hdr, pay, y in Server.val_set:
            y = y.type(torch.LongTensor)
            hdr = hdr.to(device)
            pay = pay.to(device)
            y = y.to(device)
            tmp1 = Server.HDR(hdr)
            tmp2 = Server.PAY(pay)
            tmp = torch.cat((tmp1, tmp2), 1)
            if tmp.shape[0] == 1:
                continue
            tmp = Server.SR(tmp)
            y_ = Server.Enc(tmp)
            loss = loss_fn(y_, y)
            y_ = torch.max(y_, -1)[1]
            val_acc.append(y_.eq(y.data.view_as(y_)).long().cpu().sum() / y_.shape[0])
            val_loss.append(loss.cpu().item())
            f1.append(f1_score(y.cpu(), y_.cpu(), ))
        val_loss = np.mean(val_loss); val_acc = np.mean(val_acc)
        f1 = np.mean(f1)
        print(f"Server_{e}: {train_loss} {train_acc} {val_loss} {val_acc} {time.time()} {f1}")

def Fed_test(Server: rawFedServer):
    test_loss = []; test_acc = []
    loss_fn = nn.CrossEntropyLoss()
    confussion_m = np.array([[0, 0], [0, 0]])
    Server.HDR.eval(); Server.PAY.eval(); Server.SR.eval(); Server.Enc.eval()
    for hdr, pay, y in Server.test_set:
        y = y.type(torch.LongTensor)
        hdr = hdr.to(device)
        pay = pay.to(device)
        y = y.to(device)
        tmp1 = Server.HDR(hdr)
        tmp2 = Server.PAY(pay)
        tmp = torch.cat((tmp1, tmp2), 1)
        if tmp.shape[0] == 1:
            continue
        tmp = Server.SR(tmp)
        y_ = Server.Enc(tmp)
        loss = loss_fn(y_, y)
        y_ = torch.max(y_, -1)[1]
        test_acc.append(y_.eq(y.data.view_as(y_)).long().cpu().sum() / y_.shape[0])
        test_loss.append(loss.cpu().item())
        confussion_m = np.add(confussion_m, confusion_matrix(y.cpu(), y_.cpu()))
    test_loss = np.mean(test_loss); test_acc = np.mean(test_acc)
    print(f"test: {test_loss} {test_acc}")
    print(f"confusion: {confussion_m}")
