
import torch
import torch.nn as nn
import math
import os
import sys

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, curr_path)

from yoso_kernel.yoso import YOSOAttention

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attn = YOSOAttention(config)

    def forward(self, X, mask):

        Q = self.split_heads(self.query(X))
        K = self.split_heads(self.key(X))
        V = self.split_heads(self.value(X))
        attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float()).to(X.dtype)
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
