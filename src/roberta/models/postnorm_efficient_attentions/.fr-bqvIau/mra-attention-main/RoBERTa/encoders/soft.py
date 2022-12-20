import torch
import torch.nn as nn
import math
import numpy as np

class SoftmaxFreeAttentionKernel(nn.Module):
    def __init__(self, dim, num_heads, q_len, k_len, num_landmark, conv_size, max_iter=20):
        super().__init__()

        self.head_dim = int(dim // num_heads)
        self.num_head = num_heads

        self.num_landmarks = num_landmark
        self.q_seq_len = q_len
        self.k_seq_len = k_len
        self.max_iter = max_iter

        ratio = int(self.q_seq_len // self.num_landmarks)
        self.Qlandmark_op = nn.Conv1d(self.head_dim, self.head_dim, kernel_size=ratio, stride=ratio, bias=False)
        self.Qnorm_act = nn.Sequential(nn.LayerNorm(self.head_dim), nn.GELU())

        self.use_conv = conv_size > 0
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=self.num_head, out_channels=self.num_head,
                kernel_size=(conv_size, 1), padding=(conv_size // 2, 0),
                bias=False,
                groups=self.num_head)

    def forward(self, Q, V):
        b, nhead, N, headdim, = Q.size()
        # Q: [b, num_head, N, head_dim]
        Q = Q / math.sqrt(math.sqrt(self.head_dim))
        K = Q
        
        Q_landmarks = Q.reshape(b * nhead, N, headdim).transpose(-1, -2)
        Q_landmarks = self.Qlandmark_op(Q_landmarks)
        Q_landmarks = Q_landmarks.transpose(-1, -2).reshape(b, nhead, self.num_landmarks, headdim)
        Q_landmarks = self.Qnorm_act(Q_landmarks)
        K_landmarks = Q_landmarks

        kernel_1_ = torch.cdist(Q, K_landmarks, p = 2) ** 2
        kernel_1_ = torch.exp(-kernel_1_/2)

        kernel_2_ = torch.cdist(Q_landmarks, K_landmarks, p = 2) ** 2
        kernel_2_ = torch.exp(-kernel_2_/2)

        kernel_3_ = kernel_1_.transpose(-1, -2)

        X = torch.matmul(torch.matmul(kernel_1_, self.newton_inv(kernel_2_)), torch.matmul(kernel_3_, V))

        if self.use_conv:
            X += self.conv(V)

        return X

    def newton_inv(self, mat):
        P = mat
        I = torch.eye(mat.size(-1), device=mat.device)
        alpha = 2 / (torch.max(torch.sum(mat, dim=-1)) ** 2)
        beta = 0.5
        V = alpha * P
        pnorm = torch.max(torch.sum(torch.abs(I - torch.matmul(P, V)), dim=-2))
        err_cnt = 0
        while pnorm > 1.01 and err_cnt < 10:
            alpha *= beta
            V = alpha * P
            pnorm = torch.max(torch.sum(torch.abs(I - torch.matmul(P, V)), dim=-2))
            err_cnt += 1

        for i in range(self.max_iter):
            C = 2 * V - V @ P @ V
            if C.abs().max().item() > 1e4:
                break
            V = C
        return V


class SoftmaxFreeAttention(nn.Module):
    def __init__(self, dim, num_heads, q_len, k_len, num_landmark, conv_size, max_iter=20):
        super().__init__()

        self.grad_checkpointing = True
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_head = num_heads

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.attn = SoftmaxFreeAttentionKernel(dim, num_heads, q_len, k_len, num_landmark, conv_size, max_iter)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask):

        Q = self.split_heads(self.W_q(X)) * mask[:, None, :, None]
        V = self.split_heads(self.W_v(X)) * mask[:, None, :, None]
        
        with torch.cuda.amp.autocast(enabled = False):
            attn_out = self.attn(Q.float(), V.float())
            
        attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)

        return out

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config["dim"]
        self.hidden_dim = config["hidden_dim"]

        self.mha = SoftmaxFreeAttention(
            dim = self.dim, num_heads = config["num_head"], 
            q_len = config["max_seq_len"], k_len = config["max_seq_len"], 
            num_landmark = config["num_landmarks"], conv_size = config["conv_kernel_size"]
        )

        self.dropout1 = nn.Dropout(p = config["dropout_prob"])
        self.norm1 = nn.LayerNorm(self.dim)

        self.ff = torch.nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim),
        )

        self.dropout2 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, X, mask):
        mha_out = self.norm1(X + self.dropout1(self.mha(X, mask)))
        mha_out = self.norm2(mha_out + self.dropout2(self.ff(mha_out)))
        return mha_out

class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.shared_weight = config["shared_weight"]
        self.num_layers = config["num_layers"]
        if self.shared_weight:
            self.encoder = Block(config)
        else:
            self.encoders = nn.ModuleList([Block(config) for _ in range(self.num_layers)])

    def forward(self, X, mask):
        if self.shared_weight:
            for _ in range(self.num_layers):
                X = self.encoder(X, mask)
        else:
            for encoder in self.encoders:
                X = encoder(X, mask)
        return X
