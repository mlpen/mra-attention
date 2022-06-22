import os
import torch
import argparse
import pandas as pd
import json
from collections import OrderedDict
from transformers import RobertaTokenizerFast
import math
import numpy as np

def load_model_ignore_mismatch(model, state_dict):
    model_state = model.state_dict()
    mismatched_keys = []
    for key in list(state_dict.keys()):
        if key in model_state:
            src_shape = np.asarray(list(state_dict[key].shape))
            tgt_shape = np.asarray(list(model_state[key].shape))
            if src_shape.shape[0] == tgt_shape.shape[0] and np.sum(src_shape != tgt_shape).item() == 0:
                pass
            else:
                del state_dict[key]
                mismatched_keys.append(key)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict = False)
    print(f"missing_keys = {missing_keys}")
    print(f"unexpected_keys = {unexpected_keys}")
    print(f"mismatched_keys = {mismatched_keys}")

def get_grad_norm(parameters):
    grad_norm = 0
    for p in parameters:
        if p.grad is not None:
            grad_norm += p.grad.data.norm(p = 2).item() ** 2
    grad_norm = math.sqrt(grad_norm)
    return grad_norm 

def get_grad_stat(parameters, summary):
    vectorized = []
    for p in parameters:
        if p.grad is not None:
            vectorized.append(p.grad.data.reshape(-1))
    vectorized = torch.cat(vectorized, dim = 0)
    summary["grad_mean_abs_val"] = round(vectorized.abs().mean().item(), 4)
    summary["grad_max_abs_val"] = round(vectorized.abs().max().item(), 4)
    summary["grad_norm"] = round(vectorized.norm(p = 2).item(), 4)
    summary["grad_abs_val_95"] = round(torch.quantile(vectorized, 0.95).item(), 4)
    
def compute_accumu_step(batch_size, num_gpus, inst_per_gpu):
    size_per_gpu = batch_size // num_gpus
    return max(int(math.ceil(size_per_gpu / inst_per_gpu)), 1)

def get_tokenizer(path, max_seq_len):
    tokenizer = RobertaTokenizerFast.from_pretrained(path, model_max_length = max_seq_len)
    tokenizer.model_max_length = max_seq_len
    tokenizer.init_kwargs['model_max_length'] = max_seq_len
    return tokenizer

def backward(loss, amp_scaler):
    if amp_scaler is None:
        loss.backward()
    else:
        amp_scaler.scale(loss).backward()

def optimizer_step(optimizer, lr_scheduler, amp_scaler):
    if amp_scaler is None:
        optimizer.step()
    else:
        amp_scaler.step(optimizer)
        amp_scaler.update()
    lr_scheduler.step()

def add_output_to_summary(outputs, summary):
    for key in outputs:
        if key not in summary:
            summary[key] = 0
        summary[key] = round(summary[key] + outputs[key].data.item(), 6)

def partition_inputs(inputs, num_partitions, to_cuda):
    if to_cuda:
        for key in inputs:
            inputs[key] = inputs[key].cuda()

    inputs_list = [[None, {}] for _ in range(num_partitions)]
    valid_size = None
    batch_size = None

    for key in inputs:

        if batch_size is None:
            batch_size = inputs[key].size(0)
        else:
            assert batch_size == inputs[key].size(0)

        inps = torch.chunk(inputs[key], num_partitions, dim = 0)

        if valid_size is None:
            valid_size = len(inps)
        else:
            assert valid_size == len(inps)

        for idx in range(valid_size):
            inputs_list[idx][1][key] = inps[idx]
            if inputs_list[idx][0] is None:
                inputs_list[idx][0] = inps[idx].size(0) / batch_size
            else:
                assert inputs_list[idx][0] == inps[idx].size(0) / batch_size

    return inputs_list[:valid_size]

def read_data(file):
    with open(file, "r") as f:
        data = f.read().split('\n')
    round_d = {}
    for line in data:
        try:
            values = json.loads(line.replace("'",'"'))
            round_d[values["idx"]] = values
        except Exception as e:
            print(e)
    print("Done")
    return pd.DataFrame(round_d).T

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
