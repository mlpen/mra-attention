
import os
from src.args import import_config
import argparse
import sys
import json
import torch
from src.args import import_from_string

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required = True)
    args = parser.parse_args()

    print(f"args: {args}")

    torch.cuda.set_device(0)
    config = import_config(args.config)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    config.run_name = args.config.replace(os.sep, "-")
    config.save_path = os.path.join(config.output_dir, config.run_name)

    results = import_from_string(config.profiler)(config)

    with open(f"{config.save_path}.json", 'w') as f:
        json.dump(results, f, indent = 4)

if __name__ == "__main__":
    main()
