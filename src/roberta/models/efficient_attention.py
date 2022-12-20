import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
from .postnorm import RobertaModel
from src.args import import_from_string

class EfficientPostnormRobertaModel(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        encoder_modules = []
        for module in self.modules():
            if hasattr(module, "self"):
                encoder_modules.append(module)

        for module in encoder_modules:
            module.self = import_from_string(config.attention_type)(config)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, **kwargs):
        return super().forward(input_ids, token_type_ids, position_ids, attention_mask, **kwargs)
