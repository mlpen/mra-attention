
import torch
import torch.nn as nn
import math
import sys
import os

from encoders.backbone import BackboneWrapper

def pooling(inp, mode):
    if mode == "CLS":
        pooled = inp[:, 0, :]
    elif mode == "MEAN":
        pooled = inp.mean(dim = 1)
    else:
        raise Exception()
    return pooled

def append_cls(inp, mask, vocab_size):
    batch_size = inp.size(0)
    cls_id = ((vocab_size - 1) * torch.ones(batch_size, dtype = torch.long, device = inp.device)).long()
    cls_mask = torch.ones(batch_size, dtype = torch.float, device = mask.device)
    inp = torch.cat([cls_id[:, None], inp[:, :-1]], dim = -1)
    mask = torch.cat([cls_mask[:, None], mask[:, :-1]], dim = -1)
    return inp, mask

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config["embedding_dim"] == config["dim"]

        self.dim = config["embedding_dim"]

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)

        self.position_embeddings = nn.Embedding(config["max_seq_len"], config["embedding_dim"])
        torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)

        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()

        X_token = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos

        X = self.dropout(X)

        return X
    
class SCHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config["pooling_mode"]
        self.mlpblock = nn.Sequential(
            nn.Linear(config["dim"], config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], config["num_classes"])
        )

    def forward(self, inp):
        seq_score = self.mlpblock(pooling(inp, self.pooling_mode))
        return seq_score

class ModelForSC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]
        self.pooling_mode = config["pooling_mode"]
        self.vocab_size = config["vocab_size"]
        
        self.embeddings = Embeddings(config)

        self.model = BackboneWrapper(config)

        self.seq_classifer = SCHead(config)

    def forward(self, input_ids_0, mask_0, label):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            if self.pooling_mode == "CLS":
                input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)

            token_out = self.model(self.embeddings(input_ids_0), mask_0.int())
            seq_scores = self.seq_classifer(token_out)

            seq_loss = torch.nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)
            seq_accu = (seq_scores.argmax(dim = -1) == label).to(torch.float32)
            outputs = {}
            outputs["loss"] = seq_loss
            outputs["accu"] = seq_accu

        return outputs

class SCHeadDual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config["pooling_mode"]
        self.mlpblock = nn.Sequential(
            nn.Linear(config["dim"] * 4, config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], config["num_classes"])
        )

    def forward(self, inp_0, inp_1):
        X_0 = pooling(inp_0, self.pooling_mode)
        X_1 = pooling(inp_1, self.pooling_mode)
        seq_score = self.mlpblock(torch.cat([X_0, X_1, X_0 * X_1, X_0 - X_1], dim = -1))
        return seq_score

class ModelForSCDual(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]
        self.pooling_mode = config["pooling_mode"]
        self.vocab_size = config["vocab_size"]

        self.embeddings = Embeddings(config)
        
        self.model = BackboneWrapper(config)

        self.seq_classifer = SCHeadDual(config)

    def forward(self, input_ids_0, input_ids_1, mask_0, mask_1, label):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            if self.pooling_mode == "CLS":
                input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)
                input_ids_1, mask_1 = append_cls(input_ids_1, mask_1, self.vocab_size)

            token_out_0 = self.model(self.embeddings(input_ids_0), mask_0.int())
            token_out_1 = self.model(self.embeddings(input_ids_1), mask_1.int())
            seq_scores = self.seq_classifer(token_out_0, token_out_1)

            seq_loss = torch.nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)
            seq_accu = (seq_scores.argmax(dim = -1) == label).to(torch.float32)
            outputs = {}
            outputs["loss"] = seq_loss
            outputs["accu"] = seq_accu

        return outputs
