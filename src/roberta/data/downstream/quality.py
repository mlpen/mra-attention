
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

class QualityCollator:

    def __init__(
            self,
            tokenizer,
            max_sequence_length,
            max_num_candidates,
            encode_type,
            shuffle_candidates = False,
            question_first = True,
        ):

        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = 1e9
        self.max_sequence_length = max_sequence_length
        self.max_num_candidates = max_num_candidates
        self.encode_type = encode_type
        self.shuffle_candidates = shuffle_candidates
        self.question_first = question_first

        additional_tokens = ['[question]', '[/question]', '[ent]', '[/ent]']
        tokenizer.add_tokens(additional_tokens)

    def process_one_instance(self, instance):
        shuffle_candidates = self.shuffle_candidates
        tokenizer = self.tokenizer

        q_start = tokenizer.convert_tokens_to_ids('[question]')
        q_end = tokenizer.convert_tokens_to_ids('[/question]')
        ent_start = tokenizer.convert_tokens_to_ids('[ent]')
        ent_end = tokenizer.convert_tokens_to_ids('[/ent]')

        query_tokens = [q_start] + tokenizer.encode(instance['question'])[1:-1] + [q_end]
        candidate_tokens = [
            [ent_start] + tokenizer.encode(candidate)[1:-1] + [ent_end]
            for candidate in instance['options']
        ]
        answer_index = instance['answer']

        sequence = []
        segment_ids = []
        candidate_idxes = []
        curr_segment_id = 0

        sort_order = list(range(len(candidate_tokens)))
        if shuffle_candidates:
            random.shuffle(sort_order)
            new_answer_index = sort_order.index(answer_index)
            answer_index = new_answer_index

        if self.question_first:
            sequence.extend(query_tokens)
            segment_ids.extend([curr_segment_id] * len(query_tokens))
            curr_segment_id += 1

            for k in sort_order:
                candidate_idxes.append(len(sequence))
                sequence.extend(candidate_tokens[k])
                segment_ids.extend([curr_segment_id] * len(candidate_tokens[k]))
                curr_segment_id += 1
        else:
            for k in sort_order:
                candidate_idxes.append(len(sequence))
                sequence.extend(candidate_tokens[k])
                segment_ids.extend([curr_segment_id] * len(candidate_tokens[k]))
                curr_segment_id += 1

            sequence.extend(query_tokens)
            segment_ids.extend([curr_segment_id] * len(query_tokens))
            curr_segment_id += 1

        query_mask = [1] * len(sequence)

        supports_tokens = tokenizer.encode(instance['context'])
        sequence.extend(supports_tokens)
        segment_ids.extend([curr_segment_id] * len(supports_tokens))
        curr_segment_id += 1

        assert len(sequence) == len(segment_ids)

        pos_ids = list(range(len(sequence)))
        sequence_mask = [1] * len(sequence)
        candidate_mask = [1] * len(candidate_idxes)
        query_mask = query_mask + [0] * (len(sequence) - len(query_mask))

        if self.encode_type == "original":
            segment_ids = [0] * len(sequence)

        # truncate or pad sequence to max_sequence_length
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]
            pos_ids = pos_ids[:self.max_sequence_length]
            segment_ids = segment_ids[:self.max_sequence_length]
            sequence_mask = sequence_mask[:self.max_sequence_length]
            query_mask = query_mask[:self.max_sequence_length]

        while len(sequence) < self.max_sequence_length:
            sequence.append(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
            pos_ids.append(0)
            segment_ids.append(0)
            sequence_mask.append(0)
            query_mask.append(0)

        assert len(sequence) == len(pos_ids)
        assert len(sequence) == len(segment_ids)
        assert len(sequence) == len(sequence_mask)
        assert len(sequence) == len(query_mask)

        sequence = torch.tensor(sequence, dtype = torch.long)
        pos_ids = torch.tensor(pos_ids, dtype = torch.long)
        segment_ids = torch.tensor(segment_ids, dtype = torch.long)
        sequence_mask = torch.tensor(sequence_mask, dtype = torch.long)
        query_mask = torch.tensor(query_mask, dtype = torch.long)

        # pad to max_num_candidates
        assert len(candidate_idxes) <= self.max_num_candidates
        while len(candidate_idxes) < self.max_num_candidates:
            candidate_idxes.append(0)
            candidate_mask.append(0)

        assert len(candidate_idxes) == len(candidate_mask)
        candidate_idxes = torch.tensor(candidate_idxes, dtype = torch.long)
        candidate_mask = torch.tensor(candidate_mask, dtype = torch.long)

        instance = {
            "input_ids":sequence,
            "position_ids":pos_ids,
            "token_type_ids":segment_ids,
            "attention_mask":sequence_mask,
            "importance_mask":query_mask,
            "candidate_idxes":candidate_idxes,
            "candidate_mask":candidate_mask,
            "answer_index":torch.tensor(answer_index, dtype = torch.long),
        }

        return instance

    def __call__(self, instances):
        batch = {}
        for inst in instances:
            inst = self.process_one_instance(inst)
            for key in inst:
                if key not in batch:
                    batch[key] = []
                batch[key].append(inst[key])

        batch = {key:torch.stack(batch[key], dim = 0) for key in batch}

        return batch
