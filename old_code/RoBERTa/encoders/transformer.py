
import torch
import torch.nn as nn
import math
import json
from torch.utils.checkpoint import checkpoint
import sys
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_path)

features = {}

class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["dropout_prob"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V, mask):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config["dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.attn_cpt = config["attn_cpt"] if "attn_cpt" in config else False

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.attn = SoftmaxAttention(config)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask):
        
#         features["pre_QKV"].append(X.cpu().detach().numpy())

        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))
        
#         features["QKV"].append({"Q":Q.cpu().detach().numpy(), "K":K.cpu().detach().numpy(), "V":V.cpu().detach().numpy()})

        with torch.cuda.amp.autocast(enabled = False):
            if self.attn_cpt:
                attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float())
            else:
                attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
                
#         features["attn"].append(attn_out.cpu().detach().numpy())
        
        attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)
#         features["post_attn"].append(out.cpu().detach().numpy())

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

        self.mha = Attention(config)

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
#         features["post_norm1"].append(mha_out.cpu().detach().numpy())
#         features["post_mlp"].append(self.ff(mha_out).cpu().detach().numpy())
        mha_out = self.norm2(mha_out + self.dropout2(self.ff(mha_out)))
#         features["post_norm2"].append(mha_out.cpu().detach().numpy())
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
#         global features
#         features["pre_QKV"] = []
#         features["QKV"] = []
#         features["attn"] = []
#         features["post_attn"] = []
#         features["post_norm1"] = []
#         features["post_mlp"] = []
#         features["post_norm2"] = []

        if self.shared_weight:
            for _ in range(self.num_layers):
                X = self.encoder(X, mask)
        else:
            for encoder in self.encoders:
                X = encoder(X, mask)
                
#         import pickle
#         with open("transformer-features.pickle", "wb") as f:
#             pickle.dump(features, f)
#         raise Exception()
        return X
