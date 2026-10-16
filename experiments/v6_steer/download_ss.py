"""D.0 — download the SteeringSafety HF dataset configs used here (judge-free ones + Alpaca) into the scratch cache.
    HF_HUB_OFFLINE=0 python -m experiments.v6_steer.download_ss"""
from datasets import load_dataset

from steer.common import hf_cache

CONFIGS = ["SaladBench", "Alpaca", "ToxiGen", "BBQ", "FaithEvalCounterfactual", "FaithEvalInconsistent",
           "FaithEvalUnanswerable", "ARC_C", "GPQA", "CMTEST", "TruthfulQA", "Twinviews"]

if __name__ == "__main__":
    for c in CONFIGS:
        d = load_dataset("WangResearchLab/SteeringSafety", name=c, cache_dir=f"{hf_cache()}/datasets")
        print(c, {k: len(v) for k, v in d.items()}, flush=True)
