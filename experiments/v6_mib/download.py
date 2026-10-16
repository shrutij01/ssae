"""Download the v6 MIB assets into $SSAE_V6_ROOT/hf (HF_HOME): models, MIB datasets, GemmaScope 16k canonical SAEs.

    python -m experiments.v6_mib.download [--llama]

GemmaScope: the canonical 16k residual SAEs for every Gemma-2-2B layer (MIB's DBM+SAE baseline loads
``gemma-scope-2b-pt-res-canonical`` / ``layer_{L}/width_16k/canonical`` for L = 0..25).
"""

from __future__ import annotations

import argparse
import os

MODELS = ["google/gemma-2-2b", "Qwen/Qwen2.5-0.5B"]
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASETS = [("mib-bench/ravel", None), ("mib-bench/copycolors_mcqa", "4_answer_choices"), ("mib-bench/arc_easy", None)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", action="store_true")
    ap.add_argument("--skip-sae", action="store_true")
    args = ap.parse_args()
    assert os.environ.get("HF_HOME", "").startswith(os.environ["SCRATCH"]), "HF_HOME must point to scratch"
    from huggingface_hub import snapshot_download

    for m in MODELS + ([LLAMA] if args.llama else []):
        p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"])
        print("model", m, p, flush=True)
    from datasets import load_dataset

    for path, name in DATASETS:
        for split in ("train", "val", "validation", "test"):
            try:
                ds = load_dataset(path, name, split=split)
                print("dataset", path, split, len(ds), flush=True)
            except Exception:
                pass
    if not args.skip_sae:
        from sae_lens import SAE

        for layer in range(26):
            sae = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical",
                                      sae_id=f"layer_{layer}/width_16k/canonical", device="cpu")
            sae = sae[0] if isinstance(sae, tuple) else sae
            print("gemmascope", layer, sae.cfg.d_sae, flush=True)
            del sae


if __name__ == "__main__":
    main()
