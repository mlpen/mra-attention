
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import namedtuple
import os
import sys
import pickle
import time

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, curr_path)

class BackboneWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model_type = config["model_type"]

        if self.model_type == "transformer":
            from transformer import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "none":
            from none import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "nystromformer":
            from nystromformer import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "linformer":
            from linformer import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "performer":
            from performer import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "longformer":
            from longformer import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "bigbird":
            from bigbird import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "reformer":
            from reformer import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "htransformer1d":
            from htransformer1d import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "scatterbrain":
            from scatterbrain import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "soft":
            from soft import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "block_sparse_v3":
            from block_sparse_v3 import Backbone
            self.backbone = Backbone(config)
        elif self.model_type == "yoso_attention":
            from yoso_attention import Backbone
            self.backbone = Backbone(config)
        else:
            raise Exception()


    def forward(self, X, mask):
        return self.backbone(X, mask)
