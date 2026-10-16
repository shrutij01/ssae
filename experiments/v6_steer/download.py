"""Downloads for B (run on a compute node; caches on scratch, never $HOME):  python -m experiments.v6_steer.download"""
import os

from huggingface_hub import snapshot_download

from steer.common import hf_cache

if __name__ == "__main__":
    for repo, kind, pats in [("meta-llama/Meta-Llama-Guard-2-8B", "model", ["*.json", "*.safetensors", "tokenizer*"])]:
        p = snapshot_download(repo, repo_type=kind, cache_dir=hf_cache(), allow_patterns=pats, max_workers=4)
        print("OK", repo, p, sorted(os.listdir(p)), flush=True)
