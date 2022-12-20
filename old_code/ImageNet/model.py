
import torch
import torch.nn as nn
import math
import sys
import os

from encoders.backbone import BackboneWrapper

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config["embedding_dim"] == config["dim"]
        assert config["max_seq_len"] == 1024

        self.dim = config["embedding_dim"]

        self.conv = conv = nn.Conv2d(3, self.dim, kernel_size = (14, 14), padding = (5, 5), stride = (7, 7))

        self.position_embeddings = nn.Embedding(1024, config["embedding_dim"])
        torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)

        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def forward(self, image):

        batch_size, _, _, _ = image.size()

        X_token = self.conv(image).reshape(batch_size, self.dim, -1).transpose(-1, -2)
        assert X_token.shape[1] == 1024

        position_ids = torch.arange(X_token.shape[1], dtype = torch.long, device = image.device)[None, :].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos

        X = self.dropout(X)

        return X, torch.ones_like(position_ids)
    
class SCHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlpblock = nn.Sequential(
            nn.Linear(config["dim"], config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], config["num_classes"])
        )

    def forward(self, inp):
        seq_score = self.mlpblock(inp.mean(dim = 1))
        return seq_score

class ModelForSC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]
        
        self.embeddings = Embeddings(config)

        self.model = BackboneWrapper(config)

        self.seq_classifer = SCHead(config)

    def forward(self, image, label):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):
            
            tokens, mask = self.embeddings(image)
            
            token_out = self.model(tokens, mask.int())
            seq_scores = self.seq_classifer(token_out)

            seq_loss = torch.nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)

            seq_top5 = torch.topk(seq_scores, k = 5, dim = -1, largest = True, sorted = True).indices
            top1_accu = (seq_top5[:, 0] == label).float()
            top5_accu = (seq_top5 == label[:, None]).float().max(dim = -1).values
            
            outputs = {}
            outputs["loss"] = seq_loss
            outputs["top1_accu"] = top1_accu
            outputs["top5_accu"] = top5_accu

        return outputs