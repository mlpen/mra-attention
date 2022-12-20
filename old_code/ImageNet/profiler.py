import os
import sys
from encoders.backbone import BackboneWrapper as Model
import torch
import copy
import time
import pandas as pd
import json
import numpy as np
import argparse

def func(model, batch_size, seq_len, training):
    if training:
        X = torch.rand((batch_size, seq_len, dim)).cuda()
        mask = torch.ones(batch_size, seq_len).int().cuda()
        out = model(X, mask)
        out.mean().backward()
    else:
        with torch.no_grad():
            X = torch.rand((batch_size, seq_len, dim)).cuda()
            mask = torch.ones(batch_size, seq_len).int().cuda()
            out = model(X, mask)

def get_time_memory(config, seq_len, training):

    batch_size = 1
    try:
        torch.cuda.reset_peak_memory_stats()
        model = Model(config).cuda()
        while (True):
            func(model, batch_size, seq_len, training)
            batch_size = batch_size * 2
    except Exception as e:
        if not str(e).startswith("CUDA out of memory"):
            raise e
    finally:
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    batch_size = batch_size // 2

    if batch_size == 0:
        return 0, None, None, None

    while True:
        try:
            model = Model(config).cuda()
            func(model, batch_size, seq_len, training)

            time_list = []
            for _ in range(num_iter):
                torch.cuda.synchronize()
                t0 = time.time()
                func(model, batch_size, seq_len, training)
                torch.cuda.synchronize()
                t1 = time.time()
                time_list.append((t1 - t0) / batch_size)

            per_inst_time_avg = np.mean(time_list) * 1000
            per_inst_time_std = np.std(time_list) * 1000
            memory_per_inst = torch.cuda.max_memory_allocated() / batch_size / 1024 / 1024

            del model
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            print(f"seq_len={seq_len}, batch_size={batch_size}")
            print(f"per_inst_time_avg={per_inst_time_avg}, per_inst_time_std={per_inst_time_std}")
            print(f"memory_per_inst={memory_per_inst}")

            return batch_size, per_inst_time_avg, per_inst_time_std, memory_per_inst

        except Exception as e:
            if not str(e).startswith("CUDA out of memory"):
                raise e
            batch_size = batch_size // 2
            if batch_size == 0:
                return 0, None, None, None


parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
args = parser.parse_args()

num_iter = 10
model_name = args.model

curr_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(curr_path, args.model, 'config.json'), 'r') as f:
    config = json.load(f)

config = config["model"]
config["mixed_precision"] = False
config["vocab_size"] = 512
# config["embedding_dim"] = 2
dim = config["dim"]
config["hidden_dim"] = 2
config["num_layers"] = 1
if "attn_cpt" in config:
    config["attn_cpt"] = False
print(config)

results = {}
batch_size, per_inst_time_avg, per_inst_time_std, memory_per_inst = get_time_memory(config, config["max_seq_len"], True)
results["train"] = {
    "batch_size":batch_size, 
    "per_inst_time_avg (ms)":per_inst_time_avg,
    "per_inst_time_std (ms)":per_inst_time_std,
    "memory_per_inst (MB)":memory_per_inst,
}
print(results)

with open(os.path.join(curr_path, args.model, 'efficiency.json'), 'w') as f:
    json.dump(results, f, indent = 4)

