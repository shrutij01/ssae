"""Download v4 external assets to scratch: GemmaScope 16k SAEs and the base model.

    python -m experiments.v4.download_assets [--skip-base]
"""

from __future__ import annotations

import argparse
import time

from huggingface_hub import hf_hub_download, snapshot_download

from experiments.v4.common import BASE_MODEL, GEMMASCOPE_L0, GEMMASCOPE_REPO, v4_root


def gemmascope_path(layer: int, l0: int):
    return v4_root() / "gemmascope" / f"layer_{layer}" / "width_16k" / f"average_l0_{l0}" / "params.npz"


def download_gemmascope() -> None:
    for layer, l0s in GEMMASCOPE_L0.items():
        for l0 in l0s:
            target = gemmascope_path(layer, l0)
            if target.is_file():
                continue
            t0 = time.time()
            hf_hub_download(
                GEMMASCOPE_REPO,
                f"layer_{layer}/width_16k/average_l0_{l0}/params.npz",
                local_dir=str(v4_root() / "gemmascope"),
            )
            print(f"gemmascope layer_{layer} l0={l0}: {time.time() - t0:.0f}s", flush=True)


def download_base_model() -> None:
    cache = v4_root() / "hf_cache"
    t0 = time.time()
    path = snapshot_download(
        BASE_MODEL,
        cache_dir=str(cache),
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"],
    )
    print(f"base model at {path}: {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-base", action="store_true")
    args = ap.parse_args()
    download_gemmascope()
    if not args.skip_base:
        download_base_model()
    print("done", flush=True)


if __name__ == "__main__":
    main()
