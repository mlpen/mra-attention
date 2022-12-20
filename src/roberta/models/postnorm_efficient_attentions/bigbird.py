from transformers.models.big_bird.modeling_big_bird import BigBirdConfig, BigBirdBlockSparseAttention, BigBirdModel
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class BigBirdAttention(BigBirdBlockSparseAttention):
    def __init__(self, config):

        bigbird_config = BigBirdConfig()
        bigbird_config.max_position_embeddings = config.max_position_embeddings
        bigbird_config.use_bias = True
        bigbird_config.attention_type = "block_sparse"
        bigbird_config.num_attention_heads = config.num_attention_heads
        bigbird_config.hidden_size = config.hidden_size
        bigbird_config.num_random_blocks = config.attention_random_block
        bigbird_config.block_size = config.attention_block_size

        self.bigbird_config = bigbird_config

        super().__init__(bigbird_config)

    def forward(self, X, attention_mask):
        blocked_encoder_mask, band_mask, from_mask, to_mask = BigBirdModel.create_masks_for_block_sparse_attn(attention_mask, self.bigbird_config.block_size)

        context_layer = super().forward(
            hidden_states = X,
            band_mask = band_mask,
            from_mask = from_mask,
            to_mask = to_mask,
            from_blocked_mask = blocked_encoder_mask,
            to_blocked_mask = blocked_encoder_mask
        )[0]

        return context_layer

    def extra_repr(self):
        return f'num_random_blocks={self.bigbird_config.num_random_blocks}, block_size={self.bigbird_config.block_size}'
