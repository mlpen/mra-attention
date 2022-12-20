import pytorch_lightning as pl
import torch
import torch.nn as nn
import os
import json
import time
from transformers import RobertaPreTrainedModel
from collections import defaultdict
from .downstream import DownstreamModelModule
import evaluate
from .metrics import Loss, Accuracy, F1
from src.utils import filter_inputs
from src.args import import_from_string

class RobertaForHotpotQA(RobertaPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.roberta = import_from_string(config.encoder_type)(config)
        self.para_classifier = self.make_classifier(config.hidden_size, 2)
        self.sent_classifier = self.make_classifier(config.hidden_size, 2)
        self.cls_classifier = self.make_classifier(config.hidden_size, 3)
        self.ans_classifier = self.make_classifier(config.hidden_size, 2)
        self.loss_fct = Loss()
        self.accu_fct = Accuracy()
        self.F1_fct = F1()
        self.post_init()

    def make_classifier(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )

    def get_loss_accu_count(self, logits, labels):
        loss, _ = self.loss_fct(logits, labels)
        accu, count = self.accu_fct(logits, labels)
        return loss, accu, count

    def forward(
        self,
        cls_label,
        para_idxes,
        para_labels,
        sent_idxes,
        sent_labels,
        ans_start_idxes,
        ans_end_idxes,
        **kwargs
    ):
        outputs = self.roberta(**filter_inputs(self.roberta.forward, kwargs))
        sequence_output = outputs[0]
        attention_mask = kwargs["attention_mask"]

        batch_idxes = torch.arange(sequence_output.shape[0], device = sequence_output.device)[:, None]

        output = {}

        para_embedding = sequence_output[batch_idxes, para_idxes, :]
        para_logits = self.para_classifier(para_embedding)
        loss, accu, count = self.get_loss_accu_count(para_logits.reshape(-1, 2), para_labels.reshape(-1))
        f1, _ = self.F1_fct(para_logits.reshape(-1, 2), para_labels.reshape(-1))
        output["para_loss"] = loss
        output["para_accu"] = accu
        output["para_f1"] = f1
        output["para_count"] = count

        sent_embedding = sequence_output[batch_idxes, sent_idxes, :]
        sent_logits = self.sent_classifier(sent_embedding)
        loss, accu, count = self.get_loss_accu_count(sent_logits.reshape(-1, 2), sent_labels.reshape(-1))
        f1, _ = self.F1_fct(sent_logits.reshape(-1, 2), sent_labels.reshape(-1))
        output["sent_loss"] = loss
        output["sent_accu"] = accu
        output["sent_f1"] = f1
        output["sent_count"] = count

        cls_embedding = sequence_output[:, 0, :]
        cls_logits = self.cls_classifier(cls_embedding)
        loss, accu, count = self.get_loss_accu_count(cls_logits, cls_label)
        output["cls_loss"] = loss
        output["cls_accu"] = accu
        output["cls_count"] = count

        ans_logits = self.ans_classifier(sequence_output)
        ans_logits = ans_logits - 1e5 * (1 - attention_mask[:, :, None].to(ans_logits.dtype))
        ans_start_logits = ans_logits[:, :, 0]
        ans_end_logits = ans_logits[:, :, 1]

        ans_start_idx, ans_end_idx = ans_start_idxes[:, 0], ans_end_idxes[:, 0]

        loss, accu, count = self.get_loss_accu_count(ans_start_logits, ans_start_idx)
        output["ans_start_loss"] = loss
        output["ans_start_accu"] = accu

        loss, accu, count = self.get_loss_accu_count(ans_end_logits, ans_end_idx)
        output["ans_end_loss"] = loss
        output["ans_end_accu"] = accu

        ans_span_logits = ans_start_logits[:, :, None] + ans_end_logits[:, None, :]
        ans_span_logits = ans_span_logits - 1e5 * torch.tril(torch.ones_like(ans_span_logits), diagonal = -1)
        ans_span = ans_span_logits.reshape(ans_span_logits.shape[0], -1).max(dim = -1).indices
        span_start_idx = torch.div(ans_span, ans_start_logits.shape[1], rounding_mode = 'floor')
        span_end_idx = ans_span % ans_start_logits.shape[1]

        output["loss"] = (output["ans_start_loss"] + output["ans_end_loss"]) / 2 + output["cls_loss"] + output["sent_loss"] + output["para_loss"]

        return output, cls_logits, span_start_idx, span_end_idx


class HotpotQAModelModule(DownstreamModelModule):
    def __init__(self, config, data_module):
        super().__init__(config, data_module)

        self.tokenizer = data_module.tokenizer
        self.model = RobertaForHotpotQA(self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if hasattr(self.config.model, "load_pretrain"):
            print(f"********* Loading pretrained weights: {self.config.model.load_pretrain}")
            checkpoint_model = import_from_string(self.config.model.load_pretrain["model_type"]).from_pretrained(self.config.model.load_pretrain["checkpoint"])
            checkpoint_model.resize_token_embeddings(len(self.tokenizer))
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_model.state_dict(), strict = False)
            print(f"missing_keys = {missing_keys}")
            print(f"unexpected_keys = {unexpected_keys}")

        self.squad_metric = evaluate.load('squad')

    def compute_metrics(self, sequence, cls_label, ans_start_idxes, ans_end_idxes, cls_logits, span_start_idx, span_end_idx, answers):
        ans_start_idx, ans_end_idx = ans_start_idxes[:, 0], ans_end_idxes[:, 0]
        references = []
        predictions = []
        for b in range(sequence.shape[0]):
            references.append({'answers': {'answer_start': [0], 'text': [answers[b]]}, 'id': str(b)})

            answer_type = cls_logits[b].argmax(dim = -1).item()
            if answer_type in [0, 1]:
                span = "yes" if answer_type == 1 else "no"
            elif answer_type == 2:
                span = self.tokenizer.decode(sequence[b, span_start_idx[b]:(span_end_idx[b] + 1)].tolist(), skip_special_tokens = True)
            else:
                raise Exception()
            predictions.append({'prediction_text': span, 'id': str(b)})

        metrics = self.squad_metric.compute(predictions = predictions, references = references)
        return torch.tensor(metrics["exact_match"], device = sequence.device), torch.tensor(metrics["f1"], device = sequence.device)

    def training_step(self, batch, batch_idx):
        output, cls_logits, span_start_idx, span_end_idx = self.model(**batch)
        for key in output:
            self.log(f"train.{key}", output[key].item(), on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        output, cls_logits, span_start_idx, span_end_idx = self.model(**batch)
        input_ids, cls_label, ans_start_idxes, ans_end_idxes, answers = batch["input_ids"], batch["cls_label"], batch["ans_start_idxes"], batch["ans_end_idxes"], batch["answers"]
        em, f1 = self.compute_metrics(input_ids, cls_label, ans_start_idxes, ans_end_idxes, cls_logits, span_start_idx, span_end_idx, answers)
        output["span_em"] = em
        output["span_f1"] = f1
        output = self.sync_dict(output)
        for key, val in output.items():
            self.log(f"val.{key}", val.item(), on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return output
