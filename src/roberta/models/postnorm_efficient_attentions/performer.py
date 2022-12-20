
import torch
import torch.nn as nn
import math
from performer_pytorch import FastAttention

class PerformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.rp_dim = config.attention_random_feature_dim
        self.attn_fn = FastAttention(dim_heads = self.head_dim, nb_features = self.rp_dim, causal = False, kernel_fn = torch.exp)

    def forward(self, Q, K, V, mask):
        return self.attn_fn(
            Q / math.sqrt(math.sqrt(self.head_dim)),
            K / math.sqrt(math.sqrt(self.head_dim)) * mask[:, None, :, None],
            V * mask[:, None, :, None])

    def extra_repr(self):
        return f'rp_dim={self.rp_dim}'


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attn = PerformerAttention(config)

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
