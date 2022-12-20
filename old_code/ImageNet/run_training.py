from model import ModelForSC
from dataset import get_train_loader, get_val_loader
import torch
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np
import argparse
import math
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
args = parser.parse_args()

curr_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(curr_path, args.model, 'config.json'), 'r') as f:
    config = json.load(f)

checkpoint_dir = os.path.join(curr_path, args.model, 'model')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

print(config, flush = True)

model_config = config["model"]

training_config = config["training"]
inst_per_gpu = config["gpu_memory"]

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, training_config], indent = 4))

train_loader = get_train_loader(training_config["batch_size"], training_config["num_workers"])
val_loader = get_val_loader(training_config["batch_size"], training_config["num_workers"])
print("num_train_batch", len(train_loader))
print("num_val_batch", len(val_loader))

model = ModelForSC(model_config)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

if "from_cp" in config:

    from_cp = os.path.join(curr_path, config["from_cp"])
    checkpoint = torch.load(from_cp, map_location = 'cpu')

    utils.load_model_ignore_mismatch(model.module, checkpoint['model_state_dict'])
    print("Model initialized", from_cp, flush = True)
    
    dump_path = os.path.join(checkpoint_dir, f"init.model")
    torch.save({
        "model_state_dict":model.module.state_dict()
    }, dump_path)

else:
    print("Model randomly initialized", flush = True)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = training_config["learning_rate"],
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = training_config["weight_decay"]
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = training_config["learning_rate"],
    pct_start = training_config["warmup"] / (len(train_loader) * training_config["epoch"]),
    anneal_strategy = training_config["lr_decay"],
    total_steps = len(train_loader) * training_config["epoch"]
)

amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None

summary = {
    component:{"t":0, "loss":[], "top1_accu":[], "top5_accu":[], "best_accu":0, "component":component} 
    for component in ["train", "dev"]
}

start_epoch = 0
inst_pass = 0
for epoch in reversed(range(training_config["epoch"])):
    checkpoint_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.cp")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
        utils.load_model_ignore_mismatch(model.module, checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint["epoch"] + 1
        summary = checkpoint["summary"]
        print("Model restored", checkpoint_path)
        break

accumu_steps = max(training_config["batch_size"] // len(device_ids) // inst_per_gpu, 1)
print(f"accumu_steps={accumu_steps}")

init_t = time.time()
log_f_path = os.path.join(checkpoint_dir, f"output.log")
log_f = open(log_f_path, "a+")
    
def main():

    for epoch in range(start_epoch, training_config["epoch"]):
        print(f"============= {epoch} =============")
        model.train()
        for train_step_idx, batch in enumerate(train_loader):
            batch = {"image":batch[0], "label":batch[1]}
            step(batch, "train", train_step_idx)
        print_summary(summary["train"], False, epoch)
        
        model.eval()
        with torch.no_grad():
            for val_step_idx, batch in enumerate(val_loader):
                batch = {"image":batch[0], "label":batch[1]}
                step(batch, "dev", val_step_idx)
        print_summary(summary["dev"], True, epoch)
        
        dump_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.cp")
        torch.save({
            "model_state_dict":model.module.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "lr_scheduler_state_dict":lr_scheduler.state_dict(),
            "epoch":epoch,
            "summary":summary
        }, dump_path)
        print(f"Dump {dump_path}", flush = True)

def step(batch, component, step_idx):
    t0 = time.time()

    optimizer.zero_grad()
    training = component == "train"
    
    for key in batch:
        batch[key] = batch[key].cuda()
    
    outputs = {}
    
    partial_inputs_list = [{} for _ in range(accumu_steps)]
    for key in batch:
        for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
            partial_inputs_list[idx][key] = inp

    for partial_inputs in partial_inputs_list:
        partial_outputs = model(**partial_inputs)
        for key in partial_outputs:
            partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
            if key not in outputs:
                outputs[key] = partial_outputs[key]
            else:
                outputs[key] += partial_outputs[key]
        if training:
            if amp_scaler is not None:
                amp_scaler.scale(partial_outputs["loss"]).backward()
            else:
                partial_outputs["loss"].backward()
    
    if training:
        if amp_scaler is not None:
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            optimizer.step()
        lr_scheduler.step()
        
    t1 = time.time()

    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = outputs["loss"].data.item()
    top1_accu = outputs["top1_accu"].data.item()
    top5_accu = outputs["top5_accu"].data.item()
    time_since_start = time.time() - init_t
    
    print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, top1_accu={top1_accu:.4f}, top5_accu={top5_accu:.4f}\t\t\t\t", end = "\r", flush = True)

    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["top1_accu"].append(top1_accu)
    summary[component]["top5_accu"].append(top5_accu)
    

def print_summary(summary, save_if_improved, epoch):
    summary["loss"] = np.mean(summary["loss"])
    summary["top1_accu"] = np.mean(summary["top1_accu"])
    summary["top5_accu"] = np.mean(summary["top5_accu"])

    print()
    if summary["top1_accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["top1_accu"]
        if save_if_improved:
            best_accu = summary["best_accu"]
            torch.save({"model_state_dict":model.module.state_dict()}, os.path.join(checkpoint_dir, f"best.model"))
            print(f"best_accu={best_accu}. Saved best model")

    summary_round = {"epoch":epoch}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key] = summary[key]
        else:
            summary_round[key] = round(summary[key], 4)

    print(summary_round, flush = True)
    log_f.write(json.dumps(summary_round, sort_keys = True) + "\n")
    log_f.flush()

    summary["t"] = 0
    summary["loss"] = []
    summary["top1_accu"] = []
    summary["top5_accu"] = []

main()