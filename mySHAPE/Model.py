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
device = "cuda" if torch.cuda.is_available() else "cpu"


class HAE(nn.Module):
    def __init__(self):
        super(HAE, self).__init__()
        self.enc_fc = nn.Linear(N_HDR_dims, N_enc_dims)
        self.enc_pwise_conv = nn.ModuleList([nn.Linear(N_enc_dims, N_enc_dims) for i in range(N_depth_H)])
        self.dec_fc = nn.Linear(N_enc_dims, N_HDR_dims)

    def Encode(self, input):
        x = input
        x = self.enc_fc(x)
        for layer in self.enc_pwise_conv:
            x = layer(x)
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
    def __init__(self, input_dim=1):
        super(PAE, self).__init__()
        self.enc_conv = nn.Conv1d(input_dim, N_enc_dims, kernel_size=4, stride=2, padding=1)
        self.act = nn.GELU()
        self.enc_convs = nn.ModuleList([
                                           nn.Sequential(
                                               nn.Conv1d(N_enc_dims, N_enc_dims, kernel_size=3, stride=1,
                                                         groups=N_enc_dims, padding=1),
                                               nn.GELU(),
                                               nn.BatchNorm1d(N_enc_dims)
                                           )
                                           for i in range(N_depth_P)
                                       ] + [
                                           nn.Sequential(
                                               nn.Conv1d(N_enc_dims, N_enc_dims, kernel_size=1),
                                               nn.GELU(),
                                               nn.BatchNorm1d(N_enc_dims, )
                                           )
                                           for i in range(N_depth_P)
                                       ])
        self.squeeze = nn.Conv1d(N_enc_dims, N_enc_dims, kernel_size=8, stride=4, padding=2)
        self.dec_fc = nn.ConvTranspose1d(N_enc_dims, input_dim, kernel_size=8, stride=8)

    def Encode(self, input):
        x = input
        x = self.enc_conv(x)
        for i in range(N_depth_P):
            tmp = x
            x = self.enc_convs[2 * i](x)
            x = x + tmp
            tmp = x
            x = self.enc_convs[2 * i + 1](x)
            x = x + tmp
        output = self.squeeze(x)
        return output

    def Decode(self, input):
        x = self.dec_fc(input)
        x = x.squeeze()
        output = x
        return output

    def forward(self, input):
        x = self.Encode(input)
        output = self.Decode(x)
        return output


class MultAttention(nn.Module):
    def __init__(self, layers, dim, num_heads=N_Atten, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(MultAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.layers = layers
        # C = dim, C = M * C1
        self.scale = qk_scale or self.head_dim ** 0.5
        self.attn_part = nn.ModuleList([
            nn.ModuleList([
                # qkv
                nn.Linear(dim, dim * 3, bias=qkv_bias),
                # fc
                nn.Linear(dim, dim),
                # ffn
                nn.Sequential(
                    nn.Linear(N_hidden, Nff, bias=True),
                    nn.ReLU(),
                    nn.Linear(Nff, N_hidden, bias=True)
                )
            ])
            for i in range(self.layers)
        ])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.BatchNorm1d(dim)
        self.maxPool = nn.MaxPool1d(dim)
        self.dense = nn.Linear(2 * N_hidden, 1)

    # 注意力机制
    def Attention(self, input: torch.Tensor):
        x = input
        for i in range(self.layers):
            tmp = x
            B, N, C = x.shape
            # qkv:[B,N,3,M,C1] → [3,B,M,N,C1], q,k,v:[B,M,N,C1]
            qkv = self.attn_part[i][0](x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            # attn = softmax((q*kT)/sqrt(c1)), attn:[B,M,N,N]
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            # x = attn*v, x:[B,M,N,C1] → [B,N,C]
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            # fc & add and norm
            x = self.attn_part[i][1](x)
            x = self.proj_drop(x)
            x = self.norm(x + tmp)
            # FFN & add and norm
            tmp = x
            x = self.FFN(x)
            x = self.norm(x + tmp)
        output = x
        return output

    def PreIntegrality(self, input: torch.Tensor):
        x = input
        x = self.Attention(x)
        max_x = self.maxPool(x)
        min_x = -self.maxPool(-x)
        x = torch.cat([max_x, min_x], 1)
        x = self.dense(x)
        output = x
        return output

    def forward(self, input: torch.Tensor):
        x = input
        x = self.Attention(x)
        output = x
        return output
