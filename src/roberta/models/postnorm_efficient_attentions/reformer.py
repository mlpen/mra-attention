
import torch
import torch.nn as nn
from transformers.models.reformer.modeling_reformer import LSHSelfAttention, ReformerConfig

class LSHAttention(LSHSelfAttention):
    def __init__(self, config):
        self.num_hash = config.attention_num_hash
        reformer_config = ReformerConfig()
        reformer_config.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        reformer_config.num_attention_heads = config.num_attention_heads
        reformer_config.attn_layers = ["lsh"]
        reformer_config.num_hashes = config.attention_num_hash
        reformer_config.is_decoder = False
        reformer_config.max_position_embeddings = config.max_position_embeddings
        reformer_config.hidden_size = config.hidden_size
        super().__init__(reformer_config)

    def forward(self, X, mask):
        return super().forward(hidden_states = X, attention_mask = mask).hidden_states

    def extra_repr(self):
        return f'num_hash={self.num_hash}'
