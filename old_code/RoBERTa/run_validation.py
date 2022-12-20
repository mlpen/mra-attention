from model_wrapper import ModelForMaskedLM
import torch
import torch.nn as nn
import time
import os
import json
import numpy as np
import argparse
import utils
import gzip
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
parser.add_argument("--checkpoint", type = str, help = "checkpoint", dest = "checkpoint", required = True)
args = parser.parse_args()

curr_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(curr_path, args.model, 'config.json'), 'r') as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = os.path.join(curr_path, args.model, 'model')
checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
dataset_path = config["val_dataset"]

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_config], indent = 4))

########################### Loading Model ###########################

model = ModelForMaskedLM(model_config)
print(model)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
utils.load_model_ignore_mismatch(model.module, checkpoint['model_state_dict'])
print("Model restored", checkpoint_path, flush = True)

model.eval()

########################### Running Model ###########################

log_f_path = os.path.join(checkpoint_dir, f"validation_output-{args.checkpoint}.log")
log_f = open(log_f_path, "w")

data_t0 = time.time()
load_path = os.path.join(curr_path, dataset_path)
with gzip.open(load_path, "rb") as f:
    cache_batch = pickle.load(f)
data_t1 = time.time()
print(f"loaded {load_path}", round(data_t1 - data_t0, 4))

batch_size = cache_batch[0]["input_ids"].shape[0]
accumu_steps = utils.compute_accumu_step(batch_size, len(device_ids), gpu_config["inst_per_gpu"])
print("accumu_steps", accumu_steps)

epoch_summary = {}

with torch.no_grad():
    for batch_idx in range(len(cache_batch)):
        t0 = time.time()
        batch = cache_batch[batch_idx]
        summary = {}
        for percent, inputs in utils.partition_inputs(batch, accumu_steps, True):
            outputs = model(**inputs)
            for key in outputs:
                outputs[key] = outputs[key].mean() * percent
            utils.add_output_to_summary(outputs, summary)
        t1 = time.time()

        for key in summary:
            if key not in epoch_summary:
                epoch_summary[key] = []
            epoch_summary[key].append(summary[key])

        summary["idx"] = batch_idx
        summary["time"] = round(t1 - t0, 4)

        if batch_idx % pretraining_config["batches_per_report"] == 0:
            print(json.dumps(summary, sort_keys = True), flush = True)

for key in epoch_summary:
    epoch_summary[key] = np.mean(epoch_summary[key])

log_f.write(json.dumps(epoch_summary, sort_keys = True) + "\n")
log_f.close()
print(json.dumps(epoch_summary, sort_keys = True), flush = True)
