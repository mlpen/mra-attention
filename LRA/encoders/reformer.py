
import torch
import torch.nn as nn
from transformers.modeling_reformer import LSHSelfAttention, ReformerConfig

class LSHAttention(LSHSelfAttention):
    def __init__(self, config, query, key, value):
        self.num_hash = config["num_hash"]
        reformer_config = ReformerConfig()
        reformer_config.attention_head_size = config["head_dim"]
        reformer_config.num_attention_heads = config["num_head"]
        reformer_config.attn_layers = ["lsh"]
        reformer_config.num_hashes = config["num_hash"]
        reformer_config.is_decoder = False
        reformer_config.max_position_embeddings = config["max_seq_len"]
        reformer_config.hidden_size = config["dim"]
        super().__init__(reformer_config)
        self.query_key.weight = query.weight
        self.value.weight = value.weight

    def forward(self, X, mask):
        return super().forward(hidden_states = X, attention_mask = mask).hidden_states

    def extra_repr(self):
        return f'num_hash={self.num_hash}'


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config["dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.attn = LSHAttention(config, self.W_q, self.W_k, self.W_v)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask):

        with torch.cuda.amp.autocast(enabled = False):
            attn_out = self.attn(X.float(), mask.float())

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

        self.norm1 = nn.LayerNorm(config["dim"])
        self.mha = Attention(config)
        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["dim"])

        self.mlpblock = nn.Sequential(
            nn.Linear(config["dim"], config["hidden_dim"]),
            nn.GELU(),
            torch.nn.Dropout(p = config["dropout_prob"]),
            nn.Linear(config["hidden_dim"], config["dim"]),
            torch.nn.Dropout(p = config["dropout_prob"])
        )

    def forward(self, X, mask):
        X = self.dropout1(self.mha(self.norm1(X), mask)) + X
        X = self.mlpblock(self.norm2(X)) + X
        return X

class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.shared_weight = config["shared_weight"]

        if self.shared_weight:
            self.encoder = Block(config)
        else:
            self.encoders = nn.ModuleList([Block(config) for _ in range(self.num_layers)])

        self.norm = nn.LayerNorm(config["dim"])

    def forward(self, X, mask):

        if self.shared_weight:
            for _ in range(self.num_layers):
                X = self.encoder(X, mask)
        else:
            for encoder in self.encoders:
                X = encoder(X, mask)

        X = self.norm(X) * mask[:, :, None]

        return X
