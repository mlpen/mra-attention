
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

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if hasattr(config, "attention_num_block"):
            self.num_block = config.attention_num_block
        elif hasattr(config, "attention_block_per_row"):
            assert config.max_position_embeddings % 32 == 0
            self.num_block = (config.max_position_embeddings // 32) * config.attention_block_per_row

        self.num_block = min(self.num_block, int((config.max_position_embeddings // 32) ** 2))
        self.approx_mode = config.attention_approx_mode
        self.initial_prior_first_n_blocks = config.attention_initial_prior_first_n_blocks if hasattr(config, "attention_initial_prior_first_n_blocks") else 0
        self.initial_prior_diagonal_n_blocks = config.attention_initial_prior_diagonal_n_blocks if hasattr(config, "attention_initial_prior_diagonal_n_blocks") else 0
        self.input_shape = config.attention_input_shape  if hasattr(config, "attention_input_shape") else None

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

        Q = self.split_heads(self.query(X))
        K = self.split_heads(self.key(X))
        V = self.split_heads(self.value(X))

        attn_out = mra2_attention(
            Q.float(), K.float(), V.float(), mask.float(), self.num_block,
            approx_mode = self.approx_mode,
            initial_prior_first_n_blocks = self.initial_prior_first_n_blocks,
            initial_prior_diagonal_n_blocks = self.initial_prior_diagonal_n_blocks
        ).to(X.dtype)

        attn_out = self.combine_heads(attn_out)

        if self.input_shape is not None:
            attn_out = attn_out.reshape(batch_size, H // 4, W // 8, 4, 8, dim)
            attn_out = attn_out.permute(0, 1, 3, 2, 4, 5)
            attn_out = attn_out.reshape(batch_size, seq_len, dim)

        return attn_out

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.all_head_size)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_attention_heads, self.attention_head_size)
        X = X.transpose(1, 2)
        return X
