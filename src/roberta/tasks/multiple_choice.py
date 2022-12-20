import pytorch_lightning as pl
import torch
import torch.nn as nn
import os
import json
import time
from transformers import RobertaPreTrainedModel
from collections import defaultdict
from .downstream import DownstreamModelModule
from .metrics import Loss, Accuracy
from src.utils import filter_inputs
from src.args import import_from_string

class RobertaForMultipleChoice(RobertaPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.roberta = import_from_string(config.encoder_type)(config)
        self.classifier = nn.Linear(config.hidden_size, 1, bias = False)
        self.loss_fct = Loss()
        self.accu_fct = Accuracy()
        self.post_init()

    def forward(
        self,
        candidate_idxes,
        candidate_mask,
        answer_index,
        **kwargs
    ):
        outputs = self.roberta(**filter_inputs(self.roberta.forward, kwargs))
        sequence_output = outputs[0]

        batch_idxes = torch.arange(sequence_output.shape[0], device = candidate_idxes.device)[:, None]
        ent_embedding = sequence_output[batch_idxes, candidate_idxes, :]

        scores = self.classifier(ent_embedding)[:, :, 0]
        scores = scores - 1000. * (1 - candidate_mask.to(scores.dtype))

        loss, _ = self.loss_fct(scores, answer_index)
        accu, count = self.accu_fct(scores, answer_index)

        output = {
            "loss":loss, "accu":accu, "count":count
        }
        return output

class MultipleChoiceModelModule(DownstreamModelModule):
    def __init__(self, config, data_module):
        super().__init__(config, data_module)

        self.tokenizer = data_module.tokenizer
        self.model = RobertaForMultipleChoice(self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if hasattr(self.config.model, "load_pretrain"):
            print(f"********* Loading pretrained weights: {self.config.model.load_pretrain}")
            checkpoint_model = import_from_string(self.config.model.load_pretrain["model_type"]).from_pretrained(self.config.model.load_pretrain["checkpoint"])
            checkpoint_model.resize_token_embeddings(len(self.tokenizer))
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_model.state_dict(), strict = False)
            print(f"missing_keys = {missing_keys}")
            print(f"unexpected_keys = {unexpected_keys}")
