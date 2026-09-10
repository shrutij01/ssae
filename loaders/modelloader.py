"""Model loading utilities."""

import os
import torch
import yaml
from box import Box
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

def load_llamascope_checkpoint():
    model_id = "fnlp/Llama3_1-8B-Base-LXR-8x"
    filename = "Llama3_1-8B-Base-L31R-8x/checkpoints/final.safetensors"
    filepath = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir="checkpoints",
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded checkpoint to: {filepath}")
    # Load the safetensor
    state_dict = load_file(filepath)
    decoder_weight = state_dict[
        "decoder.weight"
    ]  # shape: (vocab_size, d_model)
    decoder_bias = state_dict["decoder.bias"]
    encoder_weight = state_dict["encoder.weight"]
    encoder_bias = state_dict["encoder.bias"]
    print(f"decoder.weight shape: {decoder_weight.shape}")
    return (decoder_weight, decoder_bias, encoder_weight, encoder_bias)


def load_gemmascope_checkpoint(hf_token: str | None = None):
    """Load Gemmascope checkpoint for Gemma model embeddings."""
    model_id = "google/gemma-scope-2b-pt-res"
    filename = "layer_25/width_16k/average_l0_55/params.npz"
    filepath = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir="checkpoints",
        local_dir_use_symlinks=False,
        token=hf_token or os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded Gemmascope checkpoint to: {filepath}")
    # Load the npz file
    import numpy as np

    data = np.load(filepath)
    decoder_weight = torch.from_numpy(data["W_dec"]).T  # Transpose for DictLinearAE
    decoder_bias = torch.from_numpy(data["b_dec"])
    encoder_weight = torch.from_numpy(data["W_enc"]).T  # Transpose for DictLinearAE
    encoder_bias = torch.from_numpy(data["b_enc"])
    print(f"Gemmascope decoder.weight shape: {decoder_weight.shape}")
    return (decoder_weight, decoder_bias, encoder_weight, encoder_bias)


def load_ssae_models(modeldirs):
    """Load SSAE models from multiple directories → (folded) weights & biases.

    Routes through :meth:`ssae.DictLinearAE.from_pretrained` (the SSAE-owned
    canonical loader) so a ``standardize`` input norm is folded into the weights
    (decoder atoms ``= σ⊙W_d``, in RAW residual space), consistent with every
    other loader. No-op for ``layernorm``/``none``. The NaN/Inf/empty guards are
    kept, applied to the folded weights.
    """
    from ssae import DictLinearAE

    decoder_weight_matrices = []
    decoder_bias_vectors = []
    encoder_weight_matrices = []
    encoder_bias_vectors = []

    for modeldir in modeldirs:
        try:
            model = DictLinearAE.from_pretrained(modeldir)
            tensors = {
                "decoder.weight": model.decoder.weight,
                "decoder.bias": model.decoder.bias,
                "encoder.weight": model.encoder.weight,
                "encoder.bias": model.encoder.bias,
            }
            for key, tensor in tensors.items():
                if torch.isnan(tensor).any():
                    raise ValueError(f"NaN detected in {key}")
                if torch.isinf(tensor).any():
                    raise ValueError(f"Inf detected in {key}")
                if tensor.numel() == 0:
                    raise ValueError(f"Empty tensor in {key}")

            decoder_weight_matrices.append(tensors["decoder.weight"].detach().clone())
            decoder_bias_vectors.append(tensors["decoder.bias"].detach().clone())
            encoder_weight_matrices.append(tensors["encoder.weight"].detach().clone())
            encoder_bias_vectors.append(tensors["encoder.bias"].detach().clone())

        except Exception as e:
            raise ValueError(f"Error loading model from {modeldir}: {e}")

    return (
        decoder_weight_matrices,
        decoder_bias_vectors,
        encoder_weight_matrices,
        encoder_bias_vectors,
    )


def load_pythia_sae_checkpoint(layer: int = 5, hf_token: str | None = None):
    """Load Pythia SAE checkpoint from Hugging Face hub."""
    model_id = "EleutherAI/sae-pythia-70m-32k"
    filename = f"layers.{layer}/sae.safetensors"

    filepath = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir="checkpoints",
        local_dir_use_symlinks=False,
        token=hf_token or os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded Pythia SAE checkpoint to: {filepath}")

    # Load the safetensor
    state_dict = load_file(filepath)

    # Debug: print available keys
    print(f"Available keys in Pythia SAE: {list(state_dict.keys())}")

    # Handle mixed naming convention in Pythia SAE
    if "W_dec" in state_dict and "encoder.weight" in state_dict:
        # Mixed convention: decoder uses W_dec/b_dec, encoder uses encoder.weight/encoder.bias
        encoder_weight = state_dict["encoder.weight"]
        decoder_weight = state_dict["W_dec"].T  # Transpose to match our convention
        encoder_bias = state_dict["encoder.bias"]
        decoder_bias = state_dict["b_dec"]
    elif "W_enc" in state_dict:
        # Pure W_enc/W_dec convention
        encoder_weight = state_dict["W_enc"].T  # Transpose to match our convention
        decoder_weight = state_dict["W_dec"].T  # Transpose to match our convention
        encoder_bias = state_dict.get("b_enc", torch.zeros(encoder_weight.shape[0]))
        decoder_bias = state_dict.get("b_dec", torch.zeros(decoder_weight.shape[1]))
    elif "encoder.weight" in state_dict:
        # Pure encoder.weight/decoder.weight convention
        encoder_weight = state_dict["encoder.weight"]
        decoder_weight = state_dict["decoder.weight"]
        encoder_bias = state_dict.get("encoder.bias", torch.zeros(encoder_weight.shape[0]))
        decoder_bias = state_dict.get("decoder.bias", torch.zeros(decoder_weight.shape[1]))
    else:
        raise KeyError(
            f"Could not find encoder/decoder weights in SAE file. Available keys: {list(state_dict.keys())}"
        )

    print(f"Pythia SAE encoder.weight shape: {encoder_weight.shape}")
    return (decoder_weight, decoder_bias, encoder_weight, encoder_bias)


def load_model_config(modeldir: str) -> Box:
    """Load model configuration from YAML file."""
    config_path = os.path.join(modeldir, "cfg.yaml")
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return config
