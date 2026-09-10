import argparse
import datetime
import json
import os
import pdb
import random
import re
from prompt_toolkit.shortcuts.prompt import prompt
from tqdm import tqdm
import yaml


import openai
import pandas as pd
import tiktoken
from datasets import load_dataset as hf_load_dataset

import sys


from models import LargeLanguageModel
from prompts import get_generator_prompt
from utils import *


def load_jsonl(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def ensure_list(x):
    return x if isinstance(x, list) else [x]


def build_parser():
    # Data loading parameters
    parser = argparse.ArgumentParser(description="Generate reasoning data")

    parser.add_argument(
        "-run_name", type=str, default="default", help="run name for logs"
    )
    parser.add_argument(
        "-out_dir",
        type=str,
        default="results/findings/data_generation/runs/",
        help="Output Directory",
    )
    parser.add_argument(
        "-stop", type=list, default=[], help="When to stop generation"
    )
    parser.add_argument(
        "-exp_type",
        type=str,
        default="generate_ipd",
        choices=["generate_vanilla", "generate_confounding", "generate_ipd"],
        help="Which type of experiment",
    )
    parser.add_argument(
        "-prompt_type",
        type=str,
        default="wildbreak_make_safe",
        choices=[
            "vanilla_make_safe",
            "wildbreak_make_safe",
            "confounding_causal",
            "confounding_anticausal",
            "icl_coop",
            "icl_defect",
        ],
        help="prompt type",
    )
    parser.add_argument(
        "-model_type",
        type=str,
        default="vllm",
        choices=["openai", "vllm"],
        help="Which type of model to use",
    )
    parser.add_argument(
        "-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Which model to use",
    )
    parser.add_argument(
        "-port",
        type=int,
        default=8000,
        help="port of hosted model (if using vllm)",
    )
    parser.add_argument(
        "-max_tokens", type=int, default=512, help="Maximum number of tokens"
    )
    parser.add_argument(
        "-temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument(
        "-top_p",
        type=float,
        default=1.0,
        help="top what percentage of tokens to be considered",
    )  # Alter this or temp, not both
    parser.add_argument(
        "-n",
        type=int,
        default=1,
        help="number of completions to generate for each prompt",
    )
    parser.add_argument(
        "-presence_penalty",
        type=float,
        default=0.0,
        help="positive values increases model's likelihood to talk about new topics",
    )
    parser.add_argument(
        "-frequency_penalty",
        type=float,
        default=0.0,
        help="positive values decreases model's likelihood to repeat same line verbatim",
    )

    parser.add_argument(
        "-data",
        type=str,
        default="malicious_train_categories.jsonl",
        help="Seed data filename",
    )
    parser.add_argument(
        "-continue_name", type=str, default="default", help="Continuing name"
    )

    return parser


def get_model_generation(args, previous_data, model):
    data = []
    covered_ids = []
    if previous_data is not None:
        for j in range(len(previous_data)):
            data.append(
                [
                    previous_data.loc[j]["id_"],
                    previous_data.loc[j]["instruction"],
                    previous_data.loc[j]["source"],
                    previous_data.loc[j]["category"],
                    previous_data.loc[j]["reasoning"],
                    previous_data.loc[j]["answer"],
                ]
            )
            covered_ids.append(previous_data.loc[j]["id_"])
        print("Loaded {} instructions...".format(len(data)))
    payoffs: dict[str, int] = {}
    if args.exp_type == "generate_vanilla":
        ds = hf_load_dataset("allenai/wildjailbreak", "eval")
        adversarial_ds = ds.filter(
            lambda example: example["data_type"] == "adversarial_harmful",
            batched=False,
        )
        data = adversarial_ds["train"]
        instr_marker = ["adversarial"]
    elif args.exp_type == "generate_confounding":
        triplets_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vaw_triplets.json"
        )
        if not os.path.exists(triplets_file_path):
            raise FileNotFoundError(
                f"Please generate triplets of A, B, C first!"
            )
        triplets = load_json(triplets_file_path)
        data = triplets
        instr_marker = ["A", "B", "C"]
    elif args.exp_type == "generate_ipd":
        instr_marker = ["cooperation", "defection"]
        trajectories_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "trajectories.yaml"
        )
        with open(trajectories_file_path, "r") as f:
            data = yaml.safe_load(f)
        payoffs = {"R": 3, "S": 0, "T": 2, "P": 1}

    else:
        raise NotImplementedError

    for item in tqdm(data, desc="Processing examples"):
        instr_marker = ensure_list(instr_marker)
        params = [item[marker] for marker in instr_marker]
        prompt, sys_prompt = get_generator_prompt(
            args.prompt_type, params=params, **payoffs
        )
        og_pred = model.predict(
            prompt,
            sys_prompt,
            args.max_tokens,
            args.temperature,
            1,
            1.0,
            args.stop,
        )
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        if args.exp_type == "generate_confounding":
            instr = prompt
        elif args.exp_type == "generate_vanilla":
            instr = item["adversarial"]
        elif args.exp_type == "generate_ipd":
            if args.prompt_type == "icl_coop":
                instr = "Cooperation"
            elif args.prompt_type == "icl_defect":
                instr = "Defection"
            else:
                raise ValueError("Unknown type in IPD data")
        else:
            raise NotImplementedError
        entry = {"instruction": instr, "model_output": og_pred}
        with open(
            args.out_dir + "/paired_samples.jsonl", "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with open(args.out_dir + "/logs.txt", "a") as f:
            f.write("`Instruction: " + instr + "\n\n")
            f.write("Model output:\n\n" + og_pred + "\n\n")
            f.write(
                "------------------------------------------------------------------------\n\n"
            )


def main(args):
    _, sys_prompt = get_generator_prompt(
        args.prompt_type, params=("", "", "", "", "", "", "", "")
    )

    model = LargeLanguageModel(
        model_type=args.model_type,
        model=args.model,
        sys_prompt=sys_prompt,
        port=args.port,
    )

    if args.exp_type in [
        "generate_vanilla",
        "generate_confounding",
        "generate_ipd",
    ]:
        if args.continue_name != "default":
            args.out_dir = os.path.join(args.out_dir_name, args.continue_name)
            previous_data = pd.read_json(
                args.out_dir + "/logs.txt", lines=True
            )
        else:
            previous_data = None

        get_model_generation(args, previous_data, model)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    args.out_dir_name = args.out_dir

    cur_time = str(datetime.datetime.now())
    disp_time = cur_time.split()[0] + "-" + cur_time.split()[1].split(".")[0]

    if args.run_name == "default":
        args.run_name = (
            args.exp_type
            + "_"
            + args.prompt_type
            + "_"
            + args.model
            + "_"
            + str(args.temperature)
            + "_"
            + disp_time
            + "_"
            + str(random.randint(0, 100))
        )

    args.run_name = args.run_name.replace("/", "-")

    args.out_dir = os.path.join(args.out_dir, args.run_name)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # openai.api_key = os.getenv(" ")

    main(args)
