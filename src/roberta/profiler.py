
import os
from src.args import Options
import copy
import sys
import json
import torch
import random
import time
import numpy as np
from src.args import import_from_string
from transformers import AutoConfig, PretrainedConfig

def create_model(config):
    model_config = PretrainedConfig()
    for key, val in config.to_dict().items():
        setattr(model_config, key, val)
    return import_from_string(config.model_type)(model_config)

def func(model, batch_size, seq_len, training):
    sequence = torch.randint(0, 512, (batch_size, seq_len)).long().cuda()
    segment_ids = torch.zeros(batch_size, seq_len).long().cuda()
    pos_ids = torch.arange(seq_len)[None, :].repeat(batch_size, 1).long().cuda()
    sequence_mask = torch.ones(batch_size, seq_len).long().cuda()
    importance_mask = torch.zeros(batch_size, seq_len).long().cuda()
    importance_mask[:, 0] = 1
    label = torch.zeros(batch_size).long().cuda()
    inputs = {
        "input_ids":sequence,
        "token_type_ids":segment_ids,
        "position_ids":pos_ids,
        "attention_mask":sequence_mask,
        "importance_mask":importance_mask,
        "label":label,
    }
    if training:
        out = model(**inputs)
        out["loss"].mean().backward()
    else:
        with torch.no_grad():
            out = model(**inputs)

def get_time_memory(config, training, record_keys):

    batch_size = 1
    try:
        torch.cuda.reset_peak_memory_stats()
        model = create_model(config).cuda()
        while (True):
            func(model, batch_size, config.max_position_embeddings, training)
            batch_size = batch_size * 2
    except Exception as e:
        if not str(e).startswith("CUDA out of memory"):
            print(e)
            return 0, None, None, None
    finally:
        try:
            del model
        except:
            pass
        finally:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    batch_size = batch_size // 2

    if batch_size == 0:
        return 0, None, None, None

    while True:
        try:
            model = create_model(config).cuda()
            func(model, batch_size, config.max_position_embeddings, training)

            time_list = []
            for _ in range(config.num_profile_iterations):
                torch.cuda.synchronize()
                t0 = time.time()
                func(model, batch_size, config.max_position_embeddings, training)
                torch.cuda.synchronize()
                t1 = time.time()
                time_list.append((t1 - t0) / batch_size)

            per_inst_time_avg = np.mean(time_list) * 1000
            per_inst_time_std = np.std(time_list) * 1000
            memory_per_inst = torch.cuda.max_memory_allocated() / batch_size / 1024 / 1024

            del model
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            print(", ".join([f"config.{key}={getattr(config, key)}" for key in record_keys]))
            print(f"training={training}, sequence_length={config.max_position_embeddings}, batch_size={batch_size}")
            print(f"per_inst_time_avg={per_inst_time_avg}, per_inst_time_std={per_inst_time_std}")
            print(f"memory_per_inst={memory_per_inst}")

            return batch_size, per_inst_time_avg, per_inst_time_std, memory_per_inst

        except Exception as e:
            if not str(e).startswith("CUDA out of memory"):
                print(e)
                return 0, None, None, None
            batch_size = batch_size // 2
            if batch_size == 0:
                return 0, None, None, None

def profile(config, record_keys = []):
    singleton_config = True
    results = []
    for key, val in config.to_dict().items():
        if isinstance(val, Options):
            singleton_config = False
            for v in val.to_list():
                copied_config = copy.deepcopy(config)
                setattr(copied_config, key, v)
                results.extend(profile(copied_config, record_keys + [key]))
            break

    if not singleton_config:
        return results

    batch_size, per_inst_time_avg, per_inst_time_std, memory_per_inst = get_time_memory(config, True, record_keys)
    result = {
        "train.batch_size":batch_size,
        "train.per_inst_time_avg":per_inst_time_avg,
        "train.per_inst_time_std":per_inst_time_std,
        "train.memory_per_inst":memory_per_inst
    }
    # batch_size, per_inst_time_avg, per_inst_time_std, memory_per_inst = get_time_memory(config, False, record_keys)
    # result.update({
    #     "val.batch_size":batch_size,
    #     "val.per_inst_time_avg":per_inst_time_avg,
    #     "val.per_inst_time_std":per_inst_time_std,
    #     "val.memory_per_inst":memory_per_inst
    # })
    for key in record_keys:
        result[f"config.{key}"] = getattr(config, key)
    return [result]
