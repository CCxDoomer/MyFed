import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Nb = 784
Np = 32
BATCH = 16

class myPreDataSet(object):
    def __init__(self, X, Y=None, Z=None):
        self.X = X
        self.Y = Y
        if Y is None:
            self.mydata = X
        elif Z is None:
            self.mydata = [(x, y) for x, y in zip(X, Y)]
        else:
            self.mydata = [(x, y, z) for x, y, z in zip(X, Y, Z)]

    def __getitem__(self, idx):
        return self.mydata[idx]

    def __len__(self):
        return len(self.mydata)

class SM_PAY(nn.Module):
    def __init__(self):
        super(SM_PAY, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 25, 1)
        self.conv2 = nn.Conv1d(16, 32, 25, 1)
        self.maxpool = nn.MaxPool1d(3)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(76, 128)

    def forward(self, input):
        x = input
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.drop(x)
        x = self.fc(x)
        x = self.relu(x)
        output = x
        return output

class SM_HDR(nn.Module):
    def __init__(self, ):
        super(SM_HDR, self).__init__()
        self.BiGRU = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 128)

    def forward(self, input):
        x = input
        x = x.unsqueeze(1)
        _, x = self.BiGRU(x)
        x = x.transpose(0, 1)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc(x)
        x = self.relu(x)
        output = x
        return output

class SR(nn.Module):
    def __init__(self, ):
        super(SR, self).__init__()
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(128, 128)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = input
        x = self.drop(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.drop(x)
        output = x
        return output

class TS_Enc(nn.Module):
    def __init__(self):
        super(TS_Enc, self).__init__()
        self.conv = nn.Conv1d(36, 1, 1)
        self.fc1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, input):
        x = input
        x = self.conv(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.squeeze()
        x = nn.functional.log_softmax(x, dim=1)
        output = x
        return output

class Local_Loss(nn.Module):
    def __init__(self, ):
        super(Local_Loss, self).__init__()
        self.conv = nn.Conv1d(36, 1, 1)
        self.fc = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = input
        x = self.conv(x)
        x = self.relu(x)
        x = self.fc(x)
        x = x.squeeze()
        x = nn.functional.log_softmax(x, dim=1)
        output = x
        return output

class myClient:
    def __init__(self, train, val, code):
        self.HDR = SM_HDR().to(device)
        self.PAY = SM_PAY().to(device)
        self.CLoss = Local_Loss().to(device)
        self.optimizer_HDR = torch.optim.Adam(self.HDR.parameters(), lr=1e-5, )
        self.optimizer_PAY = torch.optim.Adam(self.PAY.parameters(), lr=1e-5, )
        self.optimizer_Loss = torch.optim.Adam(self.CLoss.parameters(), lr=1e-5, )
        self.train_set = DataLoader(train, batch_size=BATCH, shuffle=True)
        self.val_set = DataLoader(val, batch_size=BATCH, shuffle=True)
        self.name = code

class myEdge:
    def __init__(self, train, val, code):
        self.HDR = SM_HDR().to(device)
        self.PAY = SM_PAY().to(device)
        self.SR = SR().to(device)
        self.ELoss = Local_Loss().to(device)
        self.Client = []
        self.optimizer_HDR = torch.optim.Adam(self.HDR.parameters(), lr=1e-4, )
        self.optimizer_PAY = torch.optim.Adam(self.PAY.parameters(), lr=1e-4, )
        self.optimizer_SR = torch.optim.Adam(self.SR.parameters(), lr=1e-4, )
        self.optimizer_Loss = torch.optim.Adam(self.ELoss.parameters(), lr=1e-4, )
        self.train_set = DataLoader(train, batch_size=BATCH, shuffle=True)
        self.val_set = DataLoader(val, batch_size=BATCH, shuffle=True)
        self.name = code

class myCloud:
    def __init__(self, train, val, test):
        self.HDR = SM_HDR().to(device)
        self.PAY = SM_PAY().to(device)
        self.SR = SR().to(device)
        self.Enc = TS_Enc().to(device)
        self.Edge = []
        self.optimizer_HDR = torch.optim.Adam(self.HDR.parameters(), lr=1e-3, )
        self.optimizer_PAY = torch.optim.Adam(self.PAY.parameters(), lr=1e-3, )
        self.optimizer_SR = torch.optim.Adam(self.SR.parameters(), lr=1e-3, )
        self.optimizer_Enc = torch.optim.Adam(self.Enc.parameters(), lr=1e-3, )
        self.train_set = DataLoader(train, batch_size=BATCH, shuffle=True)
        self.val_set = DataLoader(val, batch_size=BATCH, shuffle=True)
        self.test_set = DataLoader(test, batch_size=BATCH, shuffle=True)

class rawFedClient:
    def __init__(self, train, val):
        self.HDR = SM_HDR().to(device)
        self.PAY = SM_PAY().to(device)
        self.SR = SR().to(device)
        self.Enc = TS_Enc().to(device)
        self.optimizer_HDR = torch.optim.Adam(self.HDR.parameters(), lr=1e-4, )
        self.optimizer_PAY = torch.optim.Adam(self.PAY.parameters(), lr=1e-4, )
        self.optimizer_SR = torch.optim.Adam(self.SR.parameters(), lr=1e-4, )
        self.optimizer_Enc = torch.optim.Adam(self.Enc.parameters(), lr=1e-4, )
        self.train_set = DataLoader(train, batch_size=BATCH, shuffle=True)
        self.val_set = DataLoader(val, batch_size=BATCH, shuffle=True)

class rawFedServer:
    def __init__(self, train, val, test):
        self.HDR = SM_HDR().to(device)
        self.PAY = SM_PAY().to(device)
        self.SR = SR().to(device)
        self.Enc = TS_Enc().to(device)
        self.Client = []
        self.optimizer_HDR = torch.optim.Adam(self.HDR.parameters(), lr=1e-4, )
        self.optimizer_PAY = torch.optim.Adam(self.PAY.parameters(), lr=1e-4, )
        self.optimizer_SR = torch.optim.Adam(self.SR.parameters(), lr=1e-4, )
        self.optimizer_Enc = torch.optim.Adam(self.Enc.parameters(), lr=1e-4, )
        self.train_set = DataLoader(train, batch_size=BATCH, shuffle=True)
        self.val_set = DataLoader(val, batch_size=BATCH, shuffle=True)
        self.test_set = DataLoader(test, batch_size=BATCH, shuffle=True)
