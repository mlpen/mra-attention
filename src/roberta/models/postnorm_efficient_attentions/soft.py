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
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_landmarks = config.attention_num_landmarks
        self.seq_len = config.max_position_embeddings

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attn = SoftmaxFreeAttentionKernel(
            dim = self.all_head_size,
            num_heads = self.num_attention_heads,
            q_len = self.seq_len,
            k_len = self.seq_len,
            num_landmark = self.num_landmarks,
            conv_size = config.attention_conv_kernel_size
        )

    def forward(self, X, mask):

        Q = self.split_heads(self.query(X)) * mask[:, None, :, None]
        V = self.split_heads(self.value(X)) * mask[:, None, :, None]

        attn_out = self.attn(Q.float(), V.float()).to(X.dtype)
        attn_out = self.combine_heads(attn_out)

        return attn_out

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.all_head_size)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_attention_heads, self.attention_head_size)
        X = X.transpose(1, 2)
        return X
