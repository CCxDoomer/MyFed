import torch
import torch.nn as nn

Nbp = 128
N_HDR_dims = 4
N_enc_dims = 94
N_depth_H = 4
N_depth_P = 3
N_hidden = N_emb_dims = N_enc_dims + 2
N_Atten = 6
Nff = 1
device = "cuda" #if torch.cuda.is_available() else "cpu"


class HAE(nn.Module):
    def __init__(self):
        super(HAE, self).__init__()
        self.enc_fc = nn.Linear(N_HDR_dims, N_enc_dims)
        self.enc_pwise_conv = [None for i in range(N_depth_H)]
        for i in range(N_depth_H):
            self.enc_pwise_conv[i] = nn.Linear(N_enc_dims, N_enc_dims).to(device)
        self.dec_fc = nn.Linear(N_enc_dims, N_HDR_dims)

    def Encode(self, input):
        x = input
        x = self.enc_fc(x)
        for i in range(N_depth_H):
            x = self.enc_pwise_conv[i](x)
        output = x
        return output

    def Decode(self, input):
        x = input
        output = self.dec_fc(x)
        return output

    def forward(self, input):
        x = self.Encode(input)
        x = self.Decode(x)
        output = x
        return output


class PAE(nn.Module):
    def __init__(self, input_dim=Nbp):
        super(PAE, self).__init__()
        self.enc_conv = nn.Conv1d(input_dim, N_enc_dims, kernel_size=4, stride=2)
        self.act = nn.GELU()
        self.enc_convs = []
        for i in range(N_depth_P):
            self.enc_convs.append(nn.Sequential(
                nn.Conv1d(N_enc_dims, N_enc_dims, kernel_size=3, stride=1, groups=N_enc_dims),
                nn.GELU(),
                nn.BatchNorm1d(N_enc_dims, )
            ))
        for i in range(N_depth_P):
            self.enc_convs.append(nn.Sequential(
                nn.Conv1d(N_enc_dims, N_enc_dims, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(N_enc_dims, )
            ))
        self.squeeze = nn.Conv1d(N_enc_dims, N_enc_dims, kernel_size=8, stride=4)
        self.dec_fc = nn.Conv1d(N_enc_dims, input_dim, kernel_size=8, stride=8)

    def forward(self, input):
        x = input
        x = self.enc_conv(x)
        for i in range(N_depth_P):
            tmp = x
            x = self.enc_convs[2 * i](x)
            x = x + tmp
            tmp = x
            x = self.enc_convs[2 * i + 1](x)
            x = x + tmp
        x = self.squeeze(x)
        output = x
        return output


class MultAttention(nn.Module):
    def __init__(self, dim, num_heads=N_Atten, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(MultAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # C = dim, C = M * C1
        self.scale = qk_scale or self.head_dim ** 0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.BatchNorm1d(dim, )
        self.FFN = nn.Sequential(
            nn.Linear(N_hidden, Nff, bias=True),
            nn.ReLU(),
            nn.Linear(Nff, N_hidden, bias=True)
        )


    def forward(self, input: torch.Tensor):
        x = input
        B, N, C = x.shape
        # B = batch_size
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # qkv:[B,N,3,M,C1] → [3,B,M,N,C1]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q,k,v:[B,M,N,C1]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        # attn = softmax((q*kT)/sqrt(c1)), attn:[B,M,N,N]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = attn*v, x:[B,M,N,C1] → [B,N,C]
        x = self.proj(x)
        # fc
        x = self.proj_drop(x)
        # add and norm
        x = self.norm(x + input)
        # FFN
        tmp = x
        x = self.FFN(x)
        # add and norm
        x = self.norm(x + tmp)
        output = x
        return output


class PreIntegrality(nn.Module):
    def __init__(self):
        super(PreIntegrality, self).__init__()
        self.maxPool = nn.AdaptiveMaxPool1d(1)
        self.dense = nn.Linear(2 * N_enc_dims, 1)

    def forward(self, input: torch.Tensor):
        x = input
        max_x = self.maxPool(x)
        min_x = -self.maxPool(-x)
        x = torch.cat((max_x, min_x), 0)
        x = self.dense(x)
        output = x
        return output

