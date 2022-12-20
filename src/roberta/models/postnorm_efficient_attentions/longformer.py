from transformers.models.longformer.modeling_longformer import LongformerSelfAttention, LongformerConfig
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class LongformerAttention(LongformerSelfAttention):
    def __init__(self, config):

        self.checkpoint_attention = config.checkpoint_attention
        self.attention_window_size = config.attention_window_size
        self.attention_num_global_tokens = config.attention_num_global_tokens
        longformer_config = LongformerConfig()
        longformer_config.num_attention_heads = config.num_attention_heads
        longformer_config.hidden_size = config.hidden_size
        longformer_config.attention_window = [config.attention_window_size]

        super().__init__(longformer_config, 0)

        self.query_global = self.query
        self.key_global = self.key
        self.value_global = self.value

    def _forward(self, X, attention_mask):
        attention_mask = (attention_mask - 1) * 10000
        if self.attention_num_global_tokens > 0:
            attention_mask[:, :self.attention_num_global_tokens] = 10000
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        context_layer = super().forward(
            hidden_states = X,
            attention_mask = attention_mask,
            is_index_masked = is_index_masked,
            is_index_global_attn = is_index_global_attn,
            is_global_attn = is_global_attn,
        )[0]

        return context_layer

    def forward(self, X, attention_mask):
        if self.checkpoint_attention:
            return checkpoint(self._forward, X, attention_mask)
        else:
            return self._forward(X, attention_mask)

    def extra_repr(self):
        return f'window_size={self.attention_window_size}, num_global_tokens={self.attention_num_global_tokens}'
