
import torch
import torch.nn as nn
import math
import os
import sys

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, curr_path)

from mra2_kernel.attention import mra2_attention

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config["dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

        if "num_block" in config:
            self.num_block = config["num_block"]
        elif "block_per_row" in config:
            assert config["max_seq_len"] % 32 == 0
            self.num_block = (config["max_seq_len"] // 32) * config["block_per_row"]

        self.num_block = min(self.num_block, int((config["max_seq_len"] // 32) ** 2))
        self.approx_mode = config["approx_mode"]
        self.initial_prior_first_n_blocks = config["initial_prior_first_n_blocks"] if "initial_prior_first_n_blocks" in config else 0
        self.initial_prior_diagonal_n_blocks = config["initial_prior_diagonal_n_blocks"] if "initial_prior_diagonal_n_blocks" in config else 0
        self.input_shape = config["input_shape"] if "input_shape" in config else None

    def extra_repr(self):
        rep = [
            f'num_block = {self.num_block}',
            f'approx_mode = {self.approx_mode}',
            f'initial_prior: first_n_blocks = {self.initial_prior_first_n_blocks}',
            f'initial_prior: diagonal_n_blocks = {self.initial_prior_diagonal_n_blocks}',
            f'input_shape = {self.input_shape}',
        ]
        return "\n".join(rep)

    def forward(self, X, mask):

        batch_size, seq_len, dim = X.shape

        if self.input_shape is not None:
            assert len(self.input_shape) == 2
            H, W = self.input_shape
            assert H * W == seq_len
            assert H % 4 == 0
            assert W % 8 == 0

            X = X.reshape(batch_size, H // 4, 4, W // 8, 8, dim)
            X = X.permute(0, 1, 3, 2, 4, 5)
            X = X.reshape(batch_size, seq_len, dim)
            mask = mask.reshape(batch_size, H // 4, 4, W // 8, 8)
            mask = mask.permute(0, 1, 3, 2, 4)
            mask = mask.reshape(batch_size, seq_len)

        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        with torch.cuda.amp.autocast(enabled = False):
            attn_out = mra2_attention(
                Q.float(), K.float(), V.float(), mask.float(), self.num_block,
                approx_mode = self.approx_mode,
                initial_prior_first_n_blocks = self.initial_prior_first_n_blocks,
                initial_prior_diagonal_n_blocks = self.initial_prior_diagonal_n_blocks
            )

        attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)

        if self.input_shape is not None:
            out = out.reshape(batch_size, H // 4, W // 8, 4, 8, dim)
            out = out.permute(0, 1, 3, 2, 4, 5)
            out = out.reshape(batch_size, seq_len, dim)

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
