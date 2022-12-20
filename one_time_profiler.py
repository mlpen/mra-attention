
import os
from dataclasses import dataclass
from src.args import import_config, import_from_string
import argparse
import sys
import json
import torch
import time
import numpy as np
import sys, os

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def prepare_inputs(instance, batch_size):
    processed = {}
    for key, val in instance.items():
        if isinstance(val, list):
            processed[key] = val * batch_size
        else:
            processed[key] = torch.cat([val] * batch_size, dim = 0).cuda()
    return processed

def run_model(model_func, inputs, training = True):
    if training:
        outputs = model_func(inputs, 0)
        outputs["loss"].backward()
    else:
        with torch.no_grad():
            outputs = model_func(inputs, 0)

def profile(model_func, instance, training = True, num_iterations = 10):

    batch_size = get_largest_batch_size(model_func, instance, training)
    print(f"the largest batch size = {batch_size}")

    try:
        inputs = prepare_inputs(instance, batch_size)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        run_model(model_func, inputs, training)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        time_list = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            t0 = time.time()
            run_model(model_func, inputs, training)
            torch.cuda.synchronize()
            t1 = time.time()
            time_list.append((t1 - t0) / batch_size)

        per_inst_time_avg = np.mean(time_list) * 1000
        per_inst_time_std = np.std(time_list) * 1000
        memory_per_inst = torch.cuda.max_memory_allocated() / batch_size / 1024 / 1024

        return {
            "batch_size":batch_size,
            "per_inst_time_avg":round(per_inst_time_avg, 2),
            "per_inst_time_std":round(per_inst_time_std, 2),
            "memory_per_inst":round(memory_per_inst, 2)
        }

    except Exception as e:
        if not str(e).startswith("CUDA out of memory"):
            raise e
        else:
            try:
                batch_size = batch_size // 2
                inputs = prepare_inputs(instance, batch_size)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                run_model(model_func, inputs, training)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                time_list = []
                for _ in range(num_iterations):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    run_model(model_func, inputs, training)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    time_list.append((t1 - t0) / batch_size)

                per_inst_time_avg = np.mean(time_list) * 1000
                per_inst_time_std = np.std(time_list) * 1000
                memory_per_inst = torch.cuda.max_memory_allocated() / batch_size / 1024 / 1024

                return {
                    "batch_size":batch_size,
                    "per_inst_time_avg":round(per_inst_time_avg, 2),
                    "per_inst_time_std":round(per_inst_time_std, 2),
                    "memory_per_inst":round(memory_per_inst, 2)
                }
            except Exception as e:
                return {
                    "batch_size":0,
                    "per_inst_time_avg":None,
                    "per_inst_time_std":None,
                    "memory_per_inst":None
                }


def get_largest_batch_size(model_func, instance, training = True):
    batch_size = 1
    try:
        torch.cuda.reset_peak_memory_stats()
        while (True):
            run_model(model_func, prepare_inputs(instance, batch_size), training)
            batch_size = batch_size * 2
    except Exception as e:
        if not str(e).startswith("CUDA out of memory"):
            raise e
    finally:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return max(1, batch_size // 2)

def set_checkpointing(config, value):
    key_list = []
    for key in config.to_dict():
        if "checkpoint" in key:
            key_list.append(key)
    for key in key_list:
        val = getattr(config, key)
        if isinstance(val, bool):
            setattr(config, key, value)
            print(f"setting {key} to {value}")

def create_model(config, data):
    block_print()
    model = import_from_string(config.model.pl_module)(config, data).cuda()
    enable_print()
    return model

def main(config):
    data = import_from_string(config.data.pl_module)(config)
    data.setup()
    data.trainer = dataclass()
    data.trainer.global_rank = 0
    data.trainer.world_size = 1
    data.trainer.current_epoch = 0
    dl = data.train_dataloader()
    instance_idx, instance = next(enumerate(dl))

    set_checkpointing(config.model, False)
    model = create_model(config, data)
    results = profile(model.training_step, instance, training = True)
    print("setting all gradient checkpointing False")

    if results["batch_size"] == 0:
        set_checkpointing(config.model, True)
        model = create_model(config, data)
        results = profile(model.training_step, instance, training = True)
        print("setting all gradient checkpointing True")

    print("Full Model")
    print(json.dumps(results, indent = 4))

    profile_encoder = hasattr(model, "profile_encoder")

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if not profile_encoder:
        return

    # set_checkpointing(config.model, False)
    model = create_model(config, data)
    results = profile(model.profile_encoder, instance, training = True)
    # print("setting all gradient checkpointing False")
    #
    # if results["batch_size"] == 0:
    #     set_checkpointing(config.model, True)
    #     model = create_model(config, data)
    #     results = profile(model.profile_encoder, instance, training = True)
    #     print("setting all gradient checkpointing True")

    print("Encoder")
    print(json.dumps(results, indent = 4))

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required = True)
    args = parser.parse_args()

    print(f"args: {args}")

    torch.cuda.set_device(0)
    config = import_config(args.config)
    config.config_path = args.config
    config.optimizer.batch_size = 1
    if hasattr(config.model, "load_pretrain"):
        del config.model.load_pretrain

    main(config)
