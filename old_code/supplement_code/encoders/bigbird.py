from transformers.models.big_bird.modeling_big_bird import BigBirdConfig, BigBirdBlockSparseAttention, BigBirdModel
import torch
import torch.nn as nn

class BigBirdAttention(BigBirdBlockSparseAttention):
    def __init__(self, config, query, key, value):

        bigbird_config = BigBirdConfig()
        bigbird_config.max_position_embeddings = config["max_seq_len"]
        bigbird_config.num_random_blocks = config["random_block"]
        bigbird_config.num_attention_heads = config["num_head"]
        bigbird_config.hidden_size = config["dim"]
        bigbird_config.block_size = config["block_size"]
        bigbird_config.use_bias = True
        self.block_size = config["block_size"]

        super().__init__(bigbird_config)

        self.query.weight = query.weight
        self.key.weight = key.weight
        self.value.weight = value.weight

        self.query.bias = query.bias
        self.key.bias = key.bias
        self.value.bias = value.bias

    def forward(self, X, attention_mask):
        blocked_encoder_mask, band_mask, from_mask, to_mask = BigBirdModel.create_masks_for_block_sparse_attn(attention_mask, self.block_size)
        return super().forward(hidden_states = X, band_mask = band_mask, from_mask = from_mask, to_mask = to_mask, from_blocked_mask = blocked_encoder_mask, to_blocked_mask = blocked_encoder_mask)[0]

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config["dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.attn = BigBirdAttention(config, self.W_q, self.W_k, self.W_v)

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
