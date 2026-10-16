from random import choice
import random
import torch
import transformers

import argparse
from tqdm import tqdm
import h5py
import os
import yaml

import utils.data_utils as utils
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


def _get_data_root():
    """Central data root — reads SSAE_DATA_ROOT env var."""
    from data.registry import get_data_root
    return get_data_root()


def _get_num_concepts(dataset_name):
    """Look up num_concepts from the dataset registry."""
    from data.registry import get_num_concepts
    return get_num_concepts(dataset_name)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_embeddings(
    contexts,
    model,
    tokenizer,
    layer,
    pooling_method="last_token",
    batch_size=128,
    max_seq_len=None,
):
    print(
        f"Processing {len(contexts)} context pairs ({len(contexts)*2} total texts) in batches of {batch_size}..."
    )

    # For last_token pooling, truncate from the LEFT so the true final token (the
    # one we pool) survives; right-truncation would keep [0:max_len] and pool the
    # wrong token. Only relevant when max_seq_len bounds length.
    if max_seq_len is not None and pooling_method == "last_token":
        tokenizer.truncation_side = "left"

    all_embeddings = []
    num_batches = (len(contexts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(contexts))
        batch_contexts = contexts[start_idx:end_idx]

        # Flatten batch texts
        batch_texts = []
        for context in batch_contexts:
            batch_texts.extend([context[0], context[1]])

        # Tokenize batch. truncation=True alone is a no-op when the model sets no
        # model_max_length (e.g. gemma-2) — an explicit max_length is required to
        # actually bound sequence length; without it one long text in the batch
        # blows up the attention matrix (33-57 GiB OOM on real-world prompts).
        tokens = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_seq_len,
        ).to(device)

        # Forward pass for batch
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            hidden_states = outputs["hidden_states"][layer]

            if pooling_method == "last_token":
                # For left padding, last token is always at the rightmost position
                pooled_embeddings = hidden_states[:, -1, :]
            elif pooling_method == "sum":
                # Sum pooling across sequence dimension (length-dependent; kept
                # for historical runs — prefer "mean", dataset brief item 3)
                pooled_embeddings = hidden_states.sum(dim=1)
            elif pooling_method == "mean":
                # Mean over real (non-padding) tokens.
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
                pooled_embeddings = (hidden_states * mask).sum(1) / mask.sum(1).clamp_min(1)
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")

            # Reshape to pairs and move to CPU to save GPU memory
            batch_embeddings = pooled_embeddings.view(
                -1, 2, pooled_embeddings.shape[-1]
            ).cpu()
            all_embeddings.append(batch_embeddings)

        # Clear GPU cache after each batch
        del (
            tokens,
            outputs,
            hidden_states,
            pooled_embeddings,
        )
        torch.cuda.empty_cache()

    # Concatenate all batch results
    print(f"Concatenating {len(all_embeddings)} batches...")
    embeddings = torch.cat(all_embeddings, dim=0)

    return embeddings


def extract_embeddings_single(
    texts,
    model,
    tokenizer,
    layer,
    pooling_method="last_token",
    batch_size=128,
    max_seq_len=1024,
    activation_method="ours",
):
    """
    Extract embeddings for individual texts (not pairs).

    Args:
        texts: List of text strings
        model: The model to use for embedding extraction
        tokenizer: The tokenizer to use
        layer: Which layer to extract embeddings from
        pooling_method: How to pool sequence embeddings ('last_token' or 'sum')
        batch_size: Batch size for processing
        max_seq_len: Maximum sequence length for truncation
        activation_method: Method for extracting activations:
            - 'ours': Left padding, right truncation, extract from position -1
            - 'reference': Right padding, left truncation, extract from last actual token

    Returns:
        torch.Tensor: Embeddings of shape (num_texts, embedding_dim)
    """
    print(
        f"Processing {len(texts)} individual texts in batches of {batch_size} using '{activation_method}' method..."
    )

    # Store original tokenizer settings to restore later
    original_truncation_side = getattr(tokenizer, "truncation_side", "right")
    original_padding_side = getattr(tokenizer, "padding_side", "left")

    # Set tokenizer based on activation method
    if activation_method == "reference":
        # Reference implementation: right padding, left truncation
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "right"
    else:  # "ours"
        # Our implementation: left padding, right truncation
        tokenizer.truncation_side = "right"
        tokenizer.padding_side = "left"

    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    # Pre-compute text lengths for reference method
    text_lengths = None
    if activation_method == "reference":
        print("Pre-computing text lengths...")
        text_lengths = []
        for t in texts:
            text_lengths.append(len(tokenizer(t)["input_ids"]))

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        batch_lengths = (
            text_lengths[start_idx:end_idx] if text_lengths else None
        )

        # Tokenize batch
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)

        # Forward pass for batch
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            hidden_states = outputs["hidden_states"][layer]

            if pooling_method == "last_token":
                if activation_method == "reference":
                    # Reference: Extract at the last actual token position (not padding)
                    # activation_pos = min(length - 1, max_seq_len - 1)
                    batch_emb_list = []
                    for j, length in enumerate(batch_lengths):
                        activation_pos = min(length - 1, max_seq_len - 1)
                        batch_emb_list.append(
                            hidden_states[j, activation_pos, :].unsqueeze(0)
                        )
                    pooled_embeddings = torch.cat(batch_emb_list, dim=0)
                else:  # "ours"
                    # Ours: With left padding, last token is always at position -1
                    pooled_embeddings = hidden_states[:, -1, :]
            elif pooling_method == "sum":
                # Sum pooling across sequence dimension
                pooled_embeddings = hidden_states.sum(dim=1)
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")

            # Move to CPU to save GPU memory
            batch_embeddings = pooled_embeddings.cpu()
            all_embeddings.append(batch_embeddings)

        # Clear GPU cache after each batch
        del (
            tokens,
            outputs,
            hidden_states,
            pooled_embeddings,
        )
        torch.cuda.empty_cache()

    # Restore original tokenizer settings
    tokenizer.truncation_side = original_truncation_side
    tokenizer.padding_side = original_padding_side

    # Concatenate all batch results
    print(f"Concatenating {len(all_embeddings)} batches...")
    embeddings = torch.cat(all_embeddings, dim=0)

    return embeddings


def create_random_pairs_from_embeddings(embeddings, num_pairs=50000, seed=42):
    """
    Create random pairs from embeddings and compute their differences.

    Args:
        embeddings: Tensor of shape (num_texts, embedding_dim)
        num_pairs: Number of pairs to create
        seed: Random seed for reproducibility

    Returns:
        torch.Tensor: Paired embeddings of shape (num_pairs, 2, embedding_dim)
                      where [:, 0, :] is first embedding and [:, 1, :] is second
    """
    import random

    random.seed(seed)

    num_texts = embeddings.shape[0]
    if num_texts < 2:
        raise ValueError("Need at least 2 texts to create pairs")

    # Limit num_pairs to avoid excessive computation
    max_possible_pairs = num_texts * (num_texts - 1) // 2
    num_pairs = min(num_pairs, max_possible_pairs)

    print(f"Creating {num_pairs} random pairs from {num_texts} embeddings...")

    # Generate random pair indices
    pair_indices = []
    for _ in range(num_pairs):
        i, j = random.sample(range(num_texts), 2)
        pair_indices.append((i, j))

    # Create paired embeddings tensor
    paired_embeddings = torch.zeros(num_pairs, 2, embeddings.shape[-1])
    for idx, (i, j) in enumerate(pair_indices):
        paired_embeddings[idx, 0] = embeddings[i]
        paired_embeddings[idx, 1] = embeddings[j]

    print(f"Created paired embeddings with shape {paired_embeddings.shape}")
    return paired_embeddings


def store_embeddings(
    filename,
    cfc_train,
    cfc_test,
    cfc_train_labels=None,
    cfc_test_labels=None,
    cfc_train_texts=None,
    cfc_test_texts=None,
    cfc_train_metadata=None,
    cfc_test_metadata=None,
):
    with h5py.File(filename, "w") as hdf_file:
        hdf_file.create_dataset("cfc_train", data=cfc_train)
        hdf_file.create_dataset("cfc_test", data=cfc_test)

        # Prompt text makes per-example value-factor evaluations independently
        # reproducible. Keep it optional so non-text and legacy callers retain
        # the existing compact HDF5 layout.
        string_dtype = h5py.string_dtype(encoding="utf-8")
        if cfc_train_texts is not None:
            hdf_file.create_dataset(
                "cfc_train_texts",
                data=cfc_train_texts,
                dtype=string_dtype,
            )
        if cfc_test_texts is not None and len(cfc_test_texts) > 0:
            hdf_file.create_dataset(
                "cfc_test_texts",
                data=cfc_test_texts,
                dtype=string_dtype,
            )

        if cfc_train_labels is not None:
            # Convert labels to a format suitable for HDF5 storage
            import json

            train_labels_json = [
                json.dumps(labels) for labels in cfc_train_labels
            ]
            hdf_file.create_dataset("cfc_train_labels", data=train_labels_json)

        if cfc_test_labels is not None and len(cfc_test_labels) > 0:
            import json

            test_labels_json = [
                json.dumps(labels) for labels in cfc_test_labels
            ]
            hdf_file.create_dataset("cfc_test_labels", data=test_labels_json)

        if cfc_train_metadata is not None:
            train_metadata_json = [json.dumps(row) for row in cfc_train_metadata]
            hdf_file.create_dataset(
                "cfc_train_pair_metadata", data=train_metadata_json
            )
        if cfc_test_metadata is not None and len(cfc_test_metadata) > 0:
            test_metadata_json = [json.dumps(row) for row in cfc_test_metadata]
            hdf_file.create_dataset(
                "cfc_test_pair_metadata", data=test_metadata_json
            )


def apply_quick_sampling(
    cfc_train_tuples,
    cfc_train_labels,
    cfc_train_metadata,
    use_quick,
):
    """Apply quick sampling if enabled: randomly sample 5500 pairs."""
    if use_quick and len(cfc_train_tuples) > 5500:
        print(
            f"Quick mode: randomly sampling 5500 pairs from {len(cfc_train_tuples)} available pairs..."
        )
        # Set seed for reproducibility
        random.seed(42)
        indices = random.sample(range(len(cfc_train_tuples)), 5500)
        cfc_train_tuples = [cfc_train_tuples[i] for i in indices]
        if cfc_train_labels:
            cfc_train_labels = [cfc_train_labels[i] for i in indices]
        if cfc_train_metadata:
            cfc_train_metadata = [cfc_train_metadata[i] for i in indices]
    return cfc_train_tuples, cfc_train_labels, cfc_train_metadata


def load_model_and_tokenizer(model_id):
    """Load model and tokenizer based on model_id."""
    if model_id == "meta-llama/Llama-3.1-8B-Instruct":
        try:
            # Try with torch_dtype and device_map for better memory management
            model = transformers.LlamaForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            tokenizer = transformers.LlamaTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
        except Exception as e:
            print(
                f"Failed to load with LlamaTokenizer, trying AutoTokenizer: {e}"
            )
            # Fallback to AutoTokenizer if LlamaTokenizer fails
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_name = "llama-3.1-8b-instruct"
    elif model_id == "meta-llama/Llama-3.1-8B":
        # Base model (no instruction tuning)
        try:
            model = transformers.LlamaForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            tokenizer = transformers.LlamaTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
        except Exception as e:
            print(
                f"Failed to load with LlamaTokenizer, trying AutoTokenizer: {e}"
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_name = "llama-3.1-8b"
    elif model_id == "EleutherAI/pythia-70m-deduped":
        pythia_cache_dir = os.environ.get(
            "SSAE_PYTHIA_CACHE_DIR", "./pythia-70m-deduped/step3000"
        )
        model_revision = os.environ.get("SSAE_MODEL_REVISION", "step3000")
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision=model_revision,
            cache_dir=pythia_cache_dir,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision=model_revision,
            cache_dir=pythia_cache_dir,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_name = "pythia70m"
    elif model_id == "google/gemma-2-2b-it":
        model_revision = os.environ.get("SSAE_MODEL_REVISION")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=model_revision, low_cpu_mem_usage=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=model_revision
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_name = "gemma-2-2b-it"
    elif model_id == "google/gemma-2-2b":
        # Base model
        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_name = "gemma-2-2b"
    elif model_id == "google/gemma-2-9b-it":
        # Instruction-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_name = "gemma-2-9b-it"
    elif model_id == "google/gemma-2-9b":
        # Base model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_name = "gemma-2-9b"
    elif model_id == "google/gemma-3-1b-pt":
        # Gemma 3 1B pre-trained base model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_name = "gemma-3-1b-pt"
    else:
        raise NotImplementedError(f"Model {model_id} not supported")

    return model, tokenizer, model_name


def main(args):
    # Load model and tokenizer once at the beginning
    model, tokenizer, model_name = load_model_and_tokenizer(args.model_id)

    if args.dataset == "oodprobe":
        # New approach: Load texts from zip, generate embeddings, then randomly pair them
        zip_path = os.environ.get(
            "OODPROBE_ZIP_PATH",
            os.path.join(os.path.dirname(_get_data_root()), "OODPROBE_DATA", "cleaned_data.zip"),
        )

        # Load all texts from the zip file
        print(f"Loading texts from {zip_path}...")
        all_texts = utils.load_oodprobe_texts_from_zip(
            zip_path=zip_path,
            max_texts_per_file=None,  # Load all texts
            seed=42,
        )

        if len(all_texts) == 0:
            raise ValueError("No texts loaded from the zip file")

        print(f"Loaded {len(all_texts)} texts, now extracting embeddings...")

        # Extract embeddings for all individual texts
        all_embeddings = extract_embeddings_single(
            all_texts,
            model,
            tokenizer,
            args.layer,
            args.pooling_method,
            args.batch_size,
            activation_method=args.activation_method,
        )

        print(f"Extracted embeddings with shape {all_embeddings.shape}")

        # Randomly sample pairs from embeddings
        num_train_pairs = args.num_samples if args.num_samples else 50000
        num_test_pairs = int(
            num_train_pairs * (1 - args.split) / args.split
        )  # Maintain split ratio

        # Apply quick sampling if enabled
        if args.quick:
            num_train_pairs = min(num_train_pairs, 5500)
            num_test_pairs = min(num_test_pairs, 550)

        # Create random pairs from embeddings
        total_pairs = num_train_pairs + num_test_pairs
        paired_embeddings = create_random_pairs_from_embeddings(
            all_embeddings,
            num_pairs=total_pairs,
            seed=42,
        )

        # Split into train and test
        cfc_train_embeddings = paired_embeddings[:num_train_pairs]
        cfc_test_embeddings = paired_embeddings[num_train_pairs:]

        print(
            f"Created {len(cfc_train_embeddings)} train pairs and {len(cfc_test_embeddings)} test pairs"
        )

        # Save embeddings
        directory_location = _get_data_root()
        directory_name = os.path.join(directory_location, "oodprobe")
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        dataset_name = "oodprobe_cleaned"
        filename = f"{dataset_name}_{model_name}_{args.layer}_{args.pooling_method}.h5"
        filepath = os.path.join(directory_name, filename)
        store_embeddings(filepath, cfc_train_embeddings, cfc_test_embeddings)

        config = {
            "rep_dim": cfc_train_embeddings.shape[-1],
            "dataset": dataset_name,
            "model": args.model_id,
            "layer": args.layer,
            "pooling_method": args.pooling_method,
            "num_texts": len(all_texts),
            "num_train_pairs": len(cfc_train_embeddings),
            "num_test_pairs": len(cfc_test_embeddings),
            "split": args.split,
            "data_seed": 42,
            "num_concepts": _get_num_concepts("oodprobe"),
        }

        config_filename = f"{dataset_name}_{model_name}_{args.layer}_{args.pooling_method}.yaml"
        config_filepath = os.path.join(directory_name, config_filename)

        with open(config_filepath, "w") as f:
            yaml.dump(config, f)

        print(f"Saved {dataset_name} embeddings to {filepath}")
        print(f"Saved {dataset_name} config to {config_filepath}")

        return
    else:
        # return_labels=True threads per-pair concept labels (e.g. categorical's
        # [color,shape,object]) through to the .h5 so per-concept / diff-native
        # eval can read them. Most datasets return None labels (harmless).
        (
            cfc_train_tuples,
            cfc_test_tuples,
            cfc_train_labels,
            cfc_test_labels,
            cfc_train_metadata,
            cfc_test_metadata,
        ) = utils.load_dataset(
            dataset_name=args.dataset,
            split=args.split,
            num_samples=args.num_samples,
            return_labels=True,
            return_metadata=True,
        )

        # Apply quick sampling if enabled (keeps labels aligned)
        (
            cfc_train_tuples,
            cfc_train_labels,
            cfc_train_metadata,
        ) = apply_quick_sampling(
            cfc_train_tuples,
            cfc_train_labels,
            cfc_train_metadata,
            args.quick,
        )

    print(
        f"Extracting embeddings from {len(cfc_train_tuples)} training samples..."
    )
    cfc_train_embeddings = extract_embeddings(
        cfc_train_tuples,
        model,
        tokenizer,
        args.layer,
        args.pooling_method,
        args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    if len(cfc_test_tuples) > 0:
        print(
            f"Extracting embeddings from {len(cfc_test_tuples)} test samples..."
        )
        cfc_test_embeddings = extract_embeddings(
            cfc_test_tuples,
            model,
            tokenizer,
            args.layer,
            args.pooling_method,
            args.batch_size,
            max_seq_len=args.max_seq_len,
        )
    else:
        cfc_test_embeddings = []

    directory_location = _get_data_root()
    directory_name = os.path.join(directory_location, str(args.dataset))
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    act_tag = f"_{args.activation_method}" if args.activation_method != "ours" else ""
    quick_tag = "_quick" if args.quick else ""
    file_stem = (
        f"{args.dataset}_{model_name}_{args.layer}"
        f"_{args.pooling_method}{act_tag}{quick_tag}"
    )

    embeddings_path = os.path.join(directory_name, f"{file_stem}.h5")
    config_path = os.path.join(directory_name, f"{file_stem}.yaml")

    if os.path.exists(embeddings_path):
        from datetime import datetime
        import shutil
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_h5 = os.path.join(directory_name, f"{file_stem}_backup_{ts}.h5")
        backup_yaml = os.path.join(directory_name, f"{file_stem}_backup_{ts}.yaml")
        shutil.move(embeddings_path, backup_h5)
        if os.path.exists(config_path):
            shutil.move(config_path, backup_yaml)
        print(f"Backed up existing embeddings to {os.path.basename(backup_h5)}")

    from data.registry import get_dataset_info

    preserve_pair_metadata = (
        args.dataset in {"bias-in-bios", "categorical-contrastive"}
        or get_dataset_info(args.dataset).is_counterfactual
    )
    store_embeddings(
        embeddings_path,
        cfc_train_embeddings,
        cfc_test_embeddings,
        cfc_train_labels,
        cfc_test_labels,
        cfc_train_tuples if preserve_pair_metadata else None,
        cfc_test_tuples if preserve_pair_metadata else None,
        cfc_train_metadata,
        cfc_test_metadata,
    )
    num_concepts = _get_num_concepts(args.dataset)

    from datetime import datetime
    config = {
        "dataset": args.dataset,
        "training_dataset_length": len(cfc_train_embeddings),
        "test_dataset_length": len(cfc_test_embeddings),
        "rep_dim": cfc_train_embeddings[0][0].shape[0],
        "num_concepts": num_concepts,
        "model": args.model_id,
        "model_revision": os.environ.get(
            "SSAE_MODEL_REVISION",
            "step3000"
            if args.model_id == "EleutherAI/pythia-70m-deduped"
            else "main",
        ),
        "llm_layer": args.layer,
        "split": args.split,
        # Categorical sampling is performed by
        # load_categorical_contrastive(data_seed=42). Older configs recorded
        # 21 here even though it was never passed to that loader.
        "data_seed": (
            42
            if args.dataset in {"bias-in-bios", "categorical-contrastive"}
            else 21
        ),
        "pooling_method": args.pooling_method,
        "activation_method": args.activation_method,
        "num_samples": args.num_samples,
        "quick": args.quick,
        "batch_size": args.batch_size,
        "generated_at": datetime.now().isoformat(),
    }
    if args.dataset == "bias-in-bios":
        from data.bias_in_bios import (
            ENDPOINT_POOL_PROTOCOL,
            audit_pair_sample,
            load_official_splits,
            prepare_disjoint_endpoint_pools,
        )

        official_splits, source_provenance = load_official_splits()
        *_, endpoint_pool_audit = prepare_disjoint_endpoint_pools(
            official_splits["train"], official_splits["test"]
        )

        config.update(
            {
                "pair_schema": "bias-in-bios-arbitrary-pairs-v1",
                "pair_sampling": "uniform_ordered_distinct_endpoints",
                "pair_conditioning": "none",
                "endpoint_pool_protocol": ENDPOINT_POOL_PROTOCOL,
                "endpoint_pool_audit": endpoint_pool_audit,
                "train_source_split": "train",
                "test_source_split": "test",
                "factor_protocol": "endpoint_b_minus_endpoint_a",
                "source_provenance": source_provenance,
                "pair_audit": {
                    "train": audit_pair_sample(
                        cfc_train_labels, cfc_train_metadata
                    ),
                    "test": audit_pair_sample(cfc_test_labels, cfc_test_metadata),
                },
            }
        )
    with open(config_path, "w") as file:
        yaml.dump(config, file)


if __name__ == "__main__":
    from data.registry import DATASETS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        # Choices come from the registry so newly-registered datasets (e.g. the
        # gender-transfer domain/language variants) are accepted without editing
        # this list.
        choices=sorted(DATASETS),
        default="eng-french",
    )
    parser.add_argument(
        "--model_id",
        default="google/gemma-2-2b-it",
        choices=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-8B",
            "EleutherAI/pythia-70m-deduped",
            "google/gemma-2-2b-it",
            "google/gemma-3-1b-pt",
        ],
    )
    parser.add_argument("--layer", type=int, default=25, choices=range(0, 33))
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--pooling-method",
        choices=["last_token", "sum"],
        default="last_token",
        help="Pooling method for sequence embeddings: 'last_token' or 'sum'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for processing contexts (default: 128, optimized for A100-class GPUs)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Truncate texts to this many tokens. Required for models with no "
             "model_max_length (e.g. gemma-2) on long real-world prompts, else "
             "attention OOMs. None (default) = no truncation (fine for short "
             "synthetic prompts).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick mode: randomly sample 5500 training pairs (same as quick training)",
    )
    parser.add_argument(
        "--activation-method",
        choices=["ours", "reference"],
        default="ours",
        help="Method for extracting activations: 'ours' (left padding, position -1) or 'reference' (right padding, last actual token position)",
    )
    # Llama-3 has 32 layers, CAA paper showed most effective steering around the
    # 13th layer for Llama-2 with 33 layers

    args = parser.parse_args()
    main(args)
