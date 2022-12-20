from model_wrapper import ModelForMaskedLM
import torch
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np
import argparse
import utils
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import gzip

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
args = parser.parse_args()

curr_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(curr_path, args.model, 'config.json'), 'r') as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = os.path.join(curr_path, args.model, 'model')
dataset = config["dataset"]

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_config], indent = 4))

########################### Loading Model ###########################

model = ModelForMaskedLM(model_config)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

if "from_cp" in config:

    from_cp = os.path.join(curr_path, config["from_cp"])
    checkpoint = torch.load(from_cp, map_location = 'cpu')

    cp_pos_encoding = checkpoint['model_state_dict']['model.embeddings.position_embeddings.weight'].data.numpy()
    cp_max_seq_len, embedding_dim = cp_pos_encoding.shape
    assert model_config["max_seq_len"] >= (cp_max_seq_len - 2)
    assert model_config["max_seq_len"] % (cp_max_seq_len - 2) == 0
    num_copy = model_config["max_seq_len"] // (cp_max_seq_len - 2)
    pos_encoding = np.concatenate([cp_pos_encoding[:2, :]] + [cp_pos_encoding[2:, :]] * num_copy, axis = 0)
    checkpoint['model_state_dict']['model.embeddings.position_embeddings.weight'] = torch.tensor(pos_encoding, dtype = torch.float)

    utils.load_model_ignore_mismatch(model.module, checkpoint['model_state_dict'])
    print("Model initialized", from_cp, flush = True)
    
    dump_path = os.path.join(checkpoint_dir, f"init.model")
    torch.save({
        "model_state_dict":model.module.state_dict()
    }, dump_path)

elif config["from_pretrained_roberta"]:

    roberta_path = os.path.join(curr_path, "roberta-base-pretrained.pickle")
    with open(roberta_path, "rb") as f:
        weights = pickle.load(f)

    assert weights["model.embeddings.position_embeddings.weight"].shape[0] == (model_config["max_seq_len"] + 2)

    for key in weights:
        weights[key] = torch.tensor(weights[key])

    utils.load_model_ignore_mismatch(model.module, weights)
    print("Model initialized from RoBERTa", flush = True)

else:
    print("Model randomly initialized", flush = True)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = pretraining_config["learning_rate"],
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = 0.01
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = pretraining_config["learning_rate"],
    pct_start = pretraining_config["warmup"],
    anneal_strategy = "linear",
    epochs = pretraining_config["epoch"],
    steps_per_epoch = pretraining_config["batches_per_epoch"]
)

amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None

start_epoch = 0
inst_pass = 0
for epoch in reversed(range(pretraining_config["epoch"])):
    checkpoint_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.cp")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
        utils.load_model_ignore_mismatch(model.module, checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint["epoch"] + 1
        inst_pass = checkpoint["inst_pass"]
        print("Model restored", checkpoint_path)
        break

########################### Running Model ###########################

log_f = open(os.path.join(checkpoint_dir, "pretrain_output.log"), "a+")

model.train()

init_t = time.time()

for epoch in range(start_epoch, pretraining_config["epoch"]):
    
    data_t0 = time.time()
    load_path = os.path.join(curr_path, f"{dataset}/epoch-{epoch:04}.pickle.gzip")
    with gzip.open(load_path, "rb") as f:
        cache_batch = pickle.load(f)
    data_t1 = time.time()
    print(f"loaded {load_path}", round(data_t1 - data_t0, 4))
    assert len(cache_batch) == pretraining_config["batches_per_epoch"]
    
    batch_size = cache_batch[0]["input_ids"].shape[0]
    accumu_steps = utils.compute_accumu_step(batch_size, len(device_ids), gpu_config["inst_per_gpu"])
    print("accumu_steps", accumu_steps)

    for batch_idx in range(pretraining_config["batches_per_epoch"]):

        optimizer.zero_grad()

        batch = cache_batch[batch_idx]
        cache_batch[batch_idx] = None

        inst_pass += list(batch.values())[0].size(0)
        summary = {}
        
        step_t0 = time.time()

        for percent, inputs in utils.partition_inputs(batch, accumu_steps, True):
            outputs = model(**inputs)
            for key in outputs:
                outputs[key] = outputs[key].mean() * percent
            utils.backward(outputs["loss"], amp_scaler)
            utils.add_output_to_summary(outputs, summary)
        
        utils.optimizer_step(optimizer, lr_scheduler, amp_scaler)
        del batch

        step_t1 = time.time()

        summary["idx"] = epoch * pretraining_config["batches_per_epoch"] + batch_idx
        summary["batch_idx"] = batch_idx
        summary["epoch"] = epoch
        summary["time"] = round(step_t1 - step_t0, 4)
        summary["inst_pass"] = inst_pass
        summary["learning_rate"] = round(optimizer.param_groups[0]["lr"], 8)
        summary["time_since_start"] = round(time.time() - init_t, 4)

        log_f.write(json.dumps(summary, sort_keys = True) + "\n")

        if batch_idx % pretraining_config["batches_per_report"] == 0:
            print(json.dumps(summary, sort_keys = True), flush = True)
            log_f.flush()

    dump_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.cp")
    torch.save({
        "model_state_dict":model.module.state_dict()
    }, dump_path.replace(".cp", ".model"))
    torch.save({
        "model_state_dict":model.module.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "lr_scheduler_state_dict":lr_scheduler.state_dict(),
        "epoch":epoch,
        "inst_pass":inst_pass
    }, dump_path)
    print(f"Dump {dump_path}", flush = True)
