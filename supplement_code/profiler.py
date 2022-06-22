import pickle
import json
from model_wrapper import ModelForMaskedLM
from model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import time
import numpy as np
import sys

def entropy(logit):
    return - (F.softmax(logit, dim = -1) * F.log_softmax(logit, dim = -1)).sum(dim = -1).mean(dim = -1)

def get_outputs(features, model, seq_len):
    xformer_outputs = {}
    with torch.no_grad():
        for layer_idx in range(12):
            for s in [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]:
                X = torch.tensor(features["pre_QKV"][layer_idx])[:, :seq_len, :].cuda() * math.sqrt(s)
                mask = torch.ones(X.shape[0], X.shape[1]).cuda().int()
                attention_module = model.model.backbone.backbone.encoders[layer_idx].mha.cuda()
                Y = attention_module(X, mask).reshape(X.shape[0], X.shape[1], 12, 64)
                Q = torch.tensor(features["QKV"][layer_idx]["Q"])[:, :, :seq_len, :].cuda() * math.sqrt(s)
                K = torch.tensor(features["QKV"][layer_idx]["K"])[:, :, :seq_len, :].cuda() * math.sqrt(s)
                E = entropy(torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.shape[-1]))

                for head_idx in range(12):
                    for inst_idx in range(2):
                        xformer_outputs[(inst_idx, layer_idx, head_idx, s)] = {
                            "Y":Y[inst_idx, :, head_idx, :], "E":E[inst_idx, head_idx].item(),
                            "N":(Y[inst_idx, :, head_idx, :] ** 2).sum().sqrt().item()
                        }
                        
    return xformer_outputs


def measure_efficiency(config, seq_len, num_iter = 10):
    def func(model, batch_size, seq_len):
        input_ids = torch.randint(0, 512, (batch_size, seq_len)).long().cuda()
        out = model(input_ids)
        out.mean().backward()

    config = config.copy()
    config["mixed_precision"] = False
    config["vocab_size"] = 512
    config["embedding_dim"] = 2
    config["hidden_dim"] = 2
    config["num_layers"] = 1
    if "attn_cpt" in config:
        config["attn_cpt"] = False
    
    batch_size = 1
    try:
        torch.cuda.reset_peak_memory_stats()
        model = Model(config).cuda()
        while (True):
            func(model, batch_size, seq_len)
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
            func(model, batch_size, seq_len)

            time_list = []
            for _ in range(num_iter):
                torch.cuda.synchronize()
                t0 = time.time()
                func(model, batch_size, seq_len)
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
parser.add_argument("--method", type = str, help = "method", dest = "method", required = True)
parser.add_argument("--seq_len", type = int, help = "seq_len", dest = "seq_len", required = True)
parser.add_argument("--efficiency", dest = "profile_eff", action = "store_true", default = False)
args = parser.parse_args()

method = args.method
seq_len = args.seq_len

methods = {
    "transformer": [
        {"model_type":"transformer"}
    ],
    "longformer": [
        {"model_type":"longformer", "first_n_tokens": 1, "window_size": int(2 ** log_window)}
        for log_window in range(4, int(math.log2(seq_len)) - 1)
    ],
    "bigbird": [
        {"model_type":"bigbird", "block_size": int(2 ** log_block_size), "random_block": 3}
        for log_block_size in range(3, int(math.log2(seq_len)) - 3)
    ],
    "nystromformer": [
        {"model_type":"nystromformer", "num_landmarks": int(2 ** log_num_landmarks)}
        for log_num_landmarks in range(4, int(math.log2(seq_len)) - 1)
    ],
    "scatterbrain": [
        {"model_type":"scatterbrain", "window_size": int(2 ** log_window)}
        for log_window in range(4, int(math.log2(seq_len)) - 1)
    ],
    "mra-2-s":[
        {"model_type":"mra2", "block_per_row": int(2 ** log_block), "approx_mode": "sparse"}
        for log_block in range(0, int(math.log2(seq_len)) - 5)
    ],
    "mra-2":[
        {"model_type":"mra2", "block_per_row": int(2 ** log_block), "approx_mode": "full"}
        for log_block in range(0, int(math.log2(seq_len)) - 5)
    ],
    "linformer":[
        {"model_type":"linformer", "linformer_k": int(2 ** log_proj_dim)}
        for log_proj_dim in range(4, int(math.log2(seq_len)) + 1)
    ],
    "performer":[
        {"model_type":"performer", "rp_dim": int(2 ** log_proj_dim)}
        for log_proj_dim in range(4, int(math.log2(seq_len)) + 1)
    ],
    "htransformer1d":[
        {"model_type":"htransformer1d", }
    ],
}

compute_workload = {
    "transformer": lambda config:1,
    "htransformer1d": lambda config:1,
    "longformer": lambda config:(config["window_size"] + 1) / seq_len,
    "bigbird": lambda config:(config["block_size"] * 8) / seq_len,
    "nystromformer": lambda config:(config["num_landmarks"]) / seq_len,
    "mra-2-s": lambda config:(config["block_per_row"] * 32) / seq_len,
    "mra-2": lambda config:(config["block_per_row"] * 32) / seq_len,
    "scatterbrain": lambda config:(config["window_size"]) / seq_len,
    "performer": lambda config:(config["rp_dim"]) / seq_len,
    "linformer": lambda config:(config["linformer_k"]) / seq_len,
}

base_config = {
    "mixed_precision": False,
    "shared_weight": False,
    "vocab_size": 50265,
    "num_sen_type": 1,
    "max_seq_len": seq_len,
    "embedding_dim": 768,
    "dim": 768,
    "hidden_dim": 3072,
    "num_layers": 12,
    "dropout_prob": 0.1,
    "num_head": 12,
    "head_dim": 64,
    "model_type": "transformer"
}

results = []
method_configs = methods[method]

if args.profile_eff:
    for method_config in method_configs:
        print(method, method_config)

        method_config_copy = base_config.copy()
        method_config_copy.update(method_config.copy())

        try:
            batch_size, per_inst_time_avg, per_inst_time_std, memory_per_inst = measure_efficiency(method_config_copy, seq_len)
            workload = compute_workload[method](method_config)
            
            results.append({
                "config":method_config, 
                "workload":workload,
                "batch_size":batch_size,
                "per_inst_time_avg":per_inst_time_avg,
                "per_inst_time_std":per_inst_time_std,
                "memory_per_inst":memory_per_inst
            })

        except Exception as e:
#             raise e
            print(e)
            
    path = f"outputs/efficiency-{method}-{seq_len}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent = 4)

    print(f"Result output: {path}")
    
    sys.exit()


print("Loading intermediate features ...")
with open("transformer-features.pickle", "rb") as f:
    features = pickle.load(f)

model = ModelForMaskedLM(base_config)
model.eval()

print("Loading model weights ...")
checkpoint = torch.load("transformer.model", map_location = 'cpu')

state_dict = checkpoint['model_state_dict']
del state_dict["model.embeddings.position_embeddings.weight"]
for key in state_dict:
    shape = state_dict[key].shape
    if ".mha.ff.weight" in key:
        print(key)
        state_dict[key] = torch.eye(state_dict[key].shape[0])
    elif ".mha.ff.bias" in key:
        print(key)
        state_dict[key] = torch.zeros_like(state_dict[key])
        
model.load_state_dict(state_dict, strict = False)
transformer_outputs = get_outputs(features, model, seq_len)

del model
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("Computing approximation and profiling ...")

for method_config in method_configs:
    print(method, method_config)
        
    method_config_copy = base_config.copy()
    method_config_copy.update(method_config.copy())
        
    try:
        
        batch_size, per_inst_time_avg, per_inst_time_std, memory_per_inst = measure_efficiency(method_config_copy, seq_len)
        
        model = ModelForMaskedLM(method_config_copy)
        model.load_state_dict(state_dict, strict = False)
        model.eval()

        xformer_outputs = get_outputs(features, model, seq_len)

        data = {}
        for key in xformer_outputs:
            inst_idx, layer_idx, head_idx, s = key
            A = transformer_outputs[key]["Y"]
            A_hat = xformer_outputs[key]["Y"]
            error = ((A - A_hat) ** 2).sum().sqrt().item()
            
            if inst_idx not in data:
                data[inst_idx] = {}
            if layer_idx not in data[inst_idx]:
                data[inst_idx][layer_idx] = {}
            if head_idx not in data[inst_idx][layer_idx]:
                data[inst_idx][layer_idx][head_idx] = {}
                
            data[inst_idx][layer_idx][head_idx][s] = {
                "error_norm":error, "entropy":transformer_outputs[key]["E"], "output_norm":transformer_outputs[key]["N"]}
            
        workload = compute_workload[method](method_config)
        
        del model
        del xformer_outputs
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        results.append({
            "config":method_config, 
            "workload":workload,
            "batch_size":batch_size,
            "per_inst_time_avg":per_inst_time_avg,
            "per_inst_time_std":per_inst_time_std,
            "memory_per_inst":memory_per_inst,
            "approximation_data":data,
        })
        
    except Exception as e:
#         raise e
        print(e)

path = f"outputs/{method}-{seq_len}.json"
with open(path, "w") as f:
    json.dump(results, f, indent = 4)
    
print(f"Result output: {path}")
