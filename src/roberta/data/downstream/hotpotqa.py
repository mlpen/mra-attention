
import torch
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer
import random
import numpy as np
import torch

class HotpotQACollator:

    def __init__(
            self,
            tokenizer,
            max_sequence_length,
            encode_type,
            shuffle_supports = False,
            context_cls_is_important = False,
        ):

        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = 1e9
        self.max_sequence_length = max_sequence_length
        self.encode_type = encode_type
        self.shuffle_supports = shuffle_supports
        self.context_cls_is_important = context_cls_is_important

        additional_tokens = ['[cls]', '[qs]', '[qe]', '[ts]', '[te]', '[ss]']
        tokenizer.add_tokens(additional_tokens)

    def process_one_instance(self, instance):
        tokenizer = self.tokenizer

        def tokenize(string):
            while "  " in string:
                string = string.replace("  ", " ")
            return tokenizer.encode(" " + string.strip(), add_special_tokens = False)

        cls = tokenizer.convert_tokens_to_ids('[cls]')
        qs = tokenizer.convert_tokens_to_ids('[qs]')
        qe = tokenizer.convert_tokens_to_ids('[qe]')
        ts = tokenizer.convert_tokens_to_ids('[ts]')
        te = tokenizer.convert_tokens_to_ids('[te]')
        ss = tokenizer.convert_tokens_to_ids('[ss]')

        question = instance["question"]
        answer = instance["answer"]
        context = instance["context"]
        support = instance["supporting_facts"]

        support_facts = {}
        for title, sent_idx in zip(support["title"], support["sent_id"]):
            if title not in support_facts:
                support_facts[title] = []
            support_facts[title].append(sent_idx)

        sequence = [cls, qs] + tokenize(question) + [qe]
        segment_ids = [0] * len(sequence)
        importance_mask = [1] * len(sequence)

        paragraph_start_indices = []
        paragraph_relevancy_labels = []

        sentence_start_indices = []
        sentence_relevancy_labels = []

        context_start_idx = len(sequence)

        sort_order = list(range(len(context["title"])))
        if self.shuffle_supports:
            random.shuffle(sort_order)
        for k in sort_order:
            title, sentences = context["title"][k], context["sentences"][k]

            if len(sequence) >= self.max_sequence_length:
                continue

            doc_relevancy = title in support_facts

            paragraph_start_indices.append(len(sequence))
            paragraph_relevancy_labels.append(1 if doc_relevancy else 0)

            title_tokens = [ts] + tokenize(title) + [te]
            sequence.extend(title_tokens)
            if self.context_cls_is_important:
                importance_mask.extend([1] + [0] * (len(title_tokens) - 1))
            else:
                importance_mask.extend([0] * len(title_tokens))

            for sent_idx, sentence in enumerate(sentences):

                if len(sequence) >= self.max_sequence_length:
                    continue

                sent_relevancy = doc_relevancy and sent_idx in support_facts[title]

                sentence_start_indices.append(len(sequence))
                sentence_relevancy_labels.append(1 if sent_relevancy else 0)

                sentence_tokens = [ss] + tokenize(sentence)
                sequence.extend(sentence_tokens)
                if self.context_cls_is_important:
                    importance_mask.extend([1] + [0] * (len(sentence_tokens) - 1))
                else:
                    importance_mask.extend([0] * len(sentence_tokens))

            segment_ids.extend([segment_ids[-1] + 1] * (len(sequence) - len(segment_ids)))

        pos_ids = list(range(len(sequence)))
        sequence_mask = [1] * len(sequence)
        assert len(sequence) == len(segment_ids)
        assert len(sequence) == len(importance_mask)

        if self.encode_type == "original":
            segment_ids = [0] * len(sequence)

        # truncate or pad sequence to max_sequence_length
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]
            pos_ids = pos_ids[:self.max_sequence_length]
            segment_ids = segment_ids[:self.max_sequence_length]
            sequence_mask = sequence_mask[:self.max_sequence_length]
            importance_mask = importance_mask[:self.max_sequence_length]

        while len(sequence) < self.max_sequence_length:
            sequence.append(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
            pos_ids.append(0)
            segment_ids.append(0)
            sequence_mask.append(0)
            importance_mask.append(0)

        if answer in ["yes", "no"]:
            cls_label = 1 if answer == "yes" else 0
            answer_span_starts = []
            answer_span_ends = []
        else:
            cls_label = 2
            answer_span_starts = []
            answer_span_ends = []
            answer_token = tokenize(answer)
            answer = self.tokenizer.decode(answer_token, skip_special_tokens = True)
            compare_idx = 0
            for idx in range(len(sequence)):
                token = sequence[idx]
                if token == ss:
                    continue
                if token == answer_token[compare_idx]:
                    compare_idx += 1
                    if compare_idx == 1:
                        span_start = idx
                    if compare_idx == len(answer_token):
                        span_end = idx
                        answer_span_starts.append(span_start)
                        answer_span_ends.append(span_end)
                        compare_idx = 0
                else:
                    compare_idx = 0

        sequence = torch.tensor(sequence, dtype = torch.long)
        pos_ids = torch.tensor(pos_ids, dtype = torch.long)
        segment_ids = torch.tensor(segment_ids, dtype = torch.long)
        sequence_mask = torch.tensor(sequence_mask, dtype = torch.long)
        importance_mask = torch.tensor(importance_mask, dtype = torch.long)

        cls_label = torch.tensor(cls_label, dtype = torch.long)
        context_start_idx = torch.tensor(context_start_idx, dtype = torch.long)

        paragraph_start_indices = paragraph_start_indices + [0] * (32 - len(paragraph_start_indices))
        paragraph_relevancy_labels = paragraph_relevancy_labels + [-100] * (32 - len(paragraph_relevancy_labels))

        paragraph_start_indices = torch.tensor(paragraph_start_indices, dtype = torch.long)
        paragraph_relevancy_labels = torch.tensor(paragraph_relevancy_labels, dtype = torch.long)

        sentence_start_indices = sentence_start_indices + [0] * (self.max_sequence_length - len(sentence_start_indices))
        sentence_relevancy_labels = sentence_relevancy_labels + [-100] * (self.max_sequence_length - len(sentence_relevancy_labels))

        sentence_start_indices = torch.tensor(sentence_start_indices, dtype = torch.long)
        sentence_relevancy_labels = torch.tensor(sentence_relevancy_labels, dtype = torch.long)

        answer_span_starts = answer_span_starts + [-100] * (self.max_sequence_length - len(answer_span_starts))
        answer_span_ends = answer_span_ends + [-100] * (self.max_sequence_length - len(answer_span_ends))

        answer_span_starts = torch.tensor(answer_span_starts, dtype = torch.long)
        answer_span_ends = torch.tensor(answer_span_ends, dtype = torch.long)

        return {
            "input_ids":sequence,
            "position_ids":pos_ids,
            "token_type_ids":segment_ids,
            "attention_mask":sequence_mask,
            "importance_mask":importance_mask,
            "cls_label":cls_label,
            "context_start_idx":context_start_idx,
            "para_idxes":paragraph_start_indices,
            "para_labels":paragraph_relevancy_labels,
            "sent_idxes":sentence_start_indices,
            "sent_labels":sentence_relevancy_labels,
            "ans_start_idxes":answer_span_starts,
            "ans_end_idxes":answer_span_ends
        }, answer

    def __call__(self, instances):
        batch = {}
        batch_answer = []
        for inst in instances:
            inst, answer = self.process_one_instance(inst)
            for key in inst:
                if key not in batch:
                    batch[key] = []
                batch[key].append(inst[key])
            batch_answer.append(answer)

        batch = {key:torch.stack(batch[key], dim = 0) for key in batch}
        batch["answers"] = batch_answer

        return batch
