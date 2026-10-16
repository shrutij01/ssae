import torch

from datasets import load_dataset as hf_load_dataset

from data import base

import numpy as np
import random

import os
import json
import h5py
import pandas as pd
from pathlib import Path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tensorify(np_array, device):
    return torch.from_numpy(np_array).to(device).type(torch.float32)


def numpify(tensor):
    return tensor.detach().cpu().to(torch.float32).numpy()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_jsonl(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def create_random_pairs(
    sentences_with_binary_labels, data_seed=21, max_pairs=50000
):
    """Create random pairs from sentences with consistent seeding."""
    import random

    values_ordered = [
        "tense-present",
        "tense-past",
        "voice-active",
        "voice-passive",
        "domain-science",
        "domain-fantasy",
        "domain-news",
        "domain-other",
        "reading-level-high",
        "reading-level-low",
        "sentiment-positive",
        "sentiment-neutral",
        "sentiment-negative",
    ]

    random.seed(data_seed)

    num_pairs = max(len(sentences_with_binary_labels) * 2, max_pairs)
    cfc_tuples = []
    all_labels = []

    for _ in range(num_pairs):
        # Randomly select two different indices
        i, j = random.sample(range(len(sentences_with_binary_labels)), 2)

        sent1, labels1 = sentences_with_binary_labels[i]
        sent2, labels2 = sentences_with_binary_labels[j]

        cfc_tuples.append([sent1, sent2])

        # Create labels indicating differences between the pair
        pair_labels = {}
        for key in values_ordered:
            pair_labels[key] = (
                labels1[key] != labels2[key]
            )  # True if different
        all_labels.append(pair_labels)

    print(
        f"Created {len(cfc_tuples)} random pairs using data_seed={data_seed}"
    )
    return cfc_tuples, all_labels


def _is_counterfactual_dataset(dataset_name) -> bool:
    """True if ``dataset_name`` is a registered counterfactual (vocab + plan)
    dataset. Lets new registry entries flow through the generate-pairs path
    without editing the dispatch in :func:`load_dataset`."""
    try:
        from data.registry import get_dataset_info
        info = get_dataset_info(dataset_name)
        return bool(info and info.is_counterfactual)
    except Exception:
        return False


def load_dataset(
    dataset_name,
    num_samples,
    split=0.9,
    return_labels=False,
    return_metadata=False,
):
    # Per-pair concept labels (populated by datasets that have them, e.g.
    # categorical-contrastive). Defaulted so every branch defines them and
    # `return_labels=True` can thread them through to store_embeddings -> .h5.
    cfc_train_labels = None
    cfc_test_labels = None
    concept_labels_test = None
    cfc_train_metadata = None
    cfc_test_metadata = None

    from data.v4_pairs import V4_DATASETS

    if dataset_name in V4_DATASETS:
        # v4 designs (docs/v4/dataset-brief.md): materialised by
        # ``python -m experiments.v4.build_datasets``; grouped train/val/test
        # splits. The legacy two-split API returns train and test pairs.
        from data.v4_pairs import load_text_pairs

        (
            cfc_train_tuples,
            cfc_train_labels,
            cfc_train_metadata,
        ) = load_text_pairs(dataset_name, "train")
        cfc_test_tuples, cfc_test_labels, cfc_test_metadata = load_text_pairs(
            dataset_name, "test"
        )
    elif dataset_name == "truthful-qa":
        hf_dataset_name = "truthfulqa/truthful_qa"
        dataset_params = {
            "split": "validation",
            "name": "multiple_choice",
        }
        data = hf_load_dataset(hf_dataset_name, **dataset_params)
        # mc1_targets have a single correct choice and mc2_targets have
        # multiple choices that can be correct
        instruction = "True or False: "
        cfc_tuples = []
        for item in data:
            question = item["question"]
            correct_answer_index = item["mc1_targets"]["labels"].index(1)
            correct_answer_choice = item["mc1_targets"]["choices"][
                correct_answer_index
            ]
            wrong_answers_indices = [
                index
                for index, value in enumerate(item["mc1_targets"]["labels"])
                if value == 0
            ]
            wrong_answer_index = np.random.choice(wrong_answers_indices)

            # Randomly select a wrong answer
            wrong_answer_choice = item["mc1_targets"]["choices"][
                wrong_answer_index
            ]

            # Create the tuple
            correct_pair = f"{question} {instruction} {correct_answer_choice}"
            wrong_pair = f"{question} {instruction} {wrong_answer_choice}"
            cfc_tuples.append([wrong_pair, correct_pair])
        split_index = int(split * len(cfc_tuples))
        cfc_train_tuples = cfc_tuples[:split_index]
        cfc_test_tuples = cfc_tuples[split_index:]
        concept_labels_test = None
        cfc_train_labels = None
        cfc_test_labels = None
        cfc_train_labels = None
        cfc_test_labels = None
    elif dataset_name == "sycophancy" or dataset_name == "refusal":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sycopath = os.path.join(
            script_dir, "..", "data", "behavioral", dataset_name + ".json"
        )
        if not os.path.exists(sycopath):
            raise FileNotFoundError(
                f"Please download the dataset and place it in the data directory."
            )
        syco_data = load_json(sycopath)
        if dataset_name == "sycophancy":
            instruction = "For the following user request and model response, does the answer align with the user's behavior? Answer only with the option: '(A) Agree' or '(B) Disagree'.\n"
        elif dataset_name == "refusal":
            instruction = "For the following user request and model response, does the response refuse or comply with the user's request? Answer only with the option: '(A) Refuse' or '(B) Comply'.\n"
        else:
            raise ValueError(f"Dataset {dataset_name} impossible here.")
        cfc_tuples = []
        for item in syco_data:
            question = item["question"]
            matching_answer = item["answer_matching_behavior"]
            not_matching_answer = item["answer_not_matching_behavior"]

            # Create paired samples: question + each answer choice
            matching_pair = f"{instruction}\n{question}\n{matching_answer}"
            not_matching_pair = (
                f"{instruction}\n{question}\n{not_matching_answer}"
            )

            # Add as [not_matching, matching] to follow your reference format
            cfc_tuples.append([not_matching_pair, matching_pair])

        # Split into train/test
        split_index = int(split * len(cfc_tuples))
        cfc_train_tuples = cfc_tuples[:split_index]
        cfc_test_tuples = cfc_tuples[split_index:]
        concept_labels_test = None
        cfc_train_labels = None
        cfc_test_labels = None
    elif dataset_name == "bias-in-bios":
        (
            cfc_train_tuples,
            cfc_test_tuples,
            cfc_train_labels,
            cfc_test_labels,
            cfc_train_metadata,
            cfc_test_metadata,
        ) = load_biasinbios(
            split=split,
            num_samples=num_samples,
            return_labels=True,
            return_metadata=True,
        )
        concept_labels_test = None
    elif dataset_name == "categorical-contrastive":
        (
            cfc_train_tuples,
            cfc_test_tuples,
            cfc_train_labels,
            cfc_test_labels,
        ) = load_categorical_contrastive(split=split, num_samples=num_samples)
        concept_labels_test = None
    elif _is_counterfactual_dataset(dataset_name):
        # Any counterfactual dataset registered in data.registry (vocab + pair
        # plans), incl. the gender-transfer domain/language variants. Driven by
        # the registry flag rather than a hardcoded name list.
        from data.registry import (
            get_counterfactual_vocab_size,
            get_dataset_info,
        )

        if not 0.0 <= split <= 1.0:
            raise ValueError(f"split must be in [0, 1], got {split}")
        info = get_dataset_info(dataset_name)
        effective_num_samples = get_counterfactual_vocab_size(
            info, num_samples
        )
        split_index = int(split * effective_num_samples)
        if 0.0 < split < 1.0 and effective_num_samples >= 2:
            split_index = min(max(split_index, 1), effective_num_samples - 1)
        (
            cfc_train_tuples,
            cfc_test_tuples,
            cfc_train_labels,
            cfc_test_labels,
        ) = generate_counterfactual_pairs(
            dataset_name,
            split_index,
            effective_num_samples,
            return_split_labels=True,
            data_seed=21,
        )
        concept_labels_test = cfc_test_labels
    else:
        raise NotImplementedError(f"load_dataset: unknown dataset {dataset_name!r}")
    if concept_labels_test is not None:
        from data.registry import get_data_root
        directory_name = os.path.join(get_data_root(), str(dataset_name))
        os.makedirs(directory_name, exist_ok=True)
        concept_labels_test_path = os.path.join(
            directory_name, "concept_labels_test_.json"
        )
        if not os.path.exists(concept_labels_test_path):
            with open(concept_labels_test_path, "w") as file:
                json.dump(concept_labels_test, file)
            print(f"Wrote test concept labels to {concept_labels_test_path}")
    if return_metadata:
        return (
            cfc_train_tuples,
            cfc_test_tuples,
            cfc_train_labels,
            cfc_test_labels,
            cfc_train_metadata,
            cfc_test_metadata,
        )
    if return_labels:
        return cfc_train_tuples, cfc_test_tuples, cfc_train_labels, cfc_test_labels
    return cfc_train_tuples, cfc_test_tuples


def generate_counterfactual_pairs(
    dataset_name,
    split_index,
    num_samples,
    *,
    return_split_labels=False,
    data_seed=21,
):
    """Generate contrastive pairs for semi-synthetic counterfactual datasets.

    Uses ``data.registry`` to look up the pair construction plan, then
    delegates to :func:`sample_counterfactuals` for the actual sampling.

    Args:
        dataset_name: Registry key (e.g. ``"eng-french"``, ``"2-binary"``).
        split_index: Index separating train from test vocab items.
        num_samples: Total number of vocab items.

    Returns:
        By default, ``(train_pairs, test_pairs, concept_labels_test)`` for
        backward compatibility. With ``return_split_labels=True``, returns
        ``(train_pairs, test_pairs, train_labels, test_labels)``. Sampling and
        train shuffling are deterministic for ``data_seed``.
    """
    from data.registry import get_dataset_info, build_data_dicts

    info = get_dataset_info(dataset_name)
    train_dict, test_dict, train_plan, test_plan = build_data_dicts(
        info, split_index, num_samples
    )

    rng = random.Random(data_seed)
    cfc_train_tuples, train_labels = sample_counterfactuals(
        train_plan, train_dict, rng=rng
    )
    cfc_test_tuples, concept_labels_test = sample_counterfactuals(
        test_plan, test_dict, rng=rng
    )

    # Index-aligned vocabularies occasionally contain duplicate spellings at
    # different indices (for example German ``Kissen``). Remove any held-out
    # pair whose literal endpoint appeared in training so the HDF5 test split
    # is genuinely lexically disjoint rather than merely index-disjoint.
    train_endpoints = {
        str(value).strip().casefold()
        for pair in cfc_train_tuples
        for value in pair
    }
    kept_test_pairs = []
    kept_test_labels = []
    for index, pair in enumerate(cfc_test_tuples):
        if any(str(value).strip().casefold() in train_endpoints for value in pair):
            continue
        kept_test_pairs.append(pair)
        if concept_labels_test:
            kept_test_labels.append(concept_labels_test[index])
    cfc_test_tuples = kept_test_pairs
    if concept_labels_test:
        concept_labels_test = kept_test_labels

    order = list(range(len(cfc_train_tuples)))
    rng.shuffle(order)
    cfc_train_tuples = [cfc_train_tuples[index] for index in order]
    if train_labels:
        if len(train_labels) != len(order):
            raise ValueError("counterfactual train labels are not pair-aligned")
        train_labels = [train_labels[index] for index in order]

    train_labels = train_labels or None
    concept_labels_test = concept_labels_test or None
    if return_split_labels:
        return (
            cfc_train_tuples,
            cfc_test_tuples,
            train_labels,
            concept_labels_test,
        )
    return cfc_train_tuples, cfc_test_tuples, concept_labels_test


def sample_counterfactuals(plan, data_dict, *, rng=None):
    rng = rng or random
    all_pairs = []
    concept_ids = []
    for keys, n_samples in plan.items():
        if len(keys) == 2:
            src_key, tgt_key = keys
            concept_id = None
        else:
            src_key, tgt_key, concept_id = keys
        src_list = data_dict[src_key]
        tgt_list = data_dict[tgt_key]
        max_len = min(len(src_list), len(tgt_list))
        n_samples = min(n_samples, max_len)
        indices = rng.sample(range(max_len), n_samples)

        for idx in indices:
            src_word = src_list[idx]
            tgt_word = tgt_list[idx]

            if src_key == tgt_key and src_word == tgt_word:
                continue
            # Append the concept_id as part of the tuple
            if len(keys) == 3:
                all_pairs.append([src_word, tgt_word])
                label = concept_id
                if isinstance(label, str):
                    try:
                        decoded = json.loads(label)
                    except json.JSONDecodeError:
                        decoded = label
                    if isinstance(decoded, dict):
                        label = decoded
                concept_ids.append(label)
            else:
                all_pairs.append([src_word, tgt_word])
    return all_pairs, concept_ids


def load_test_data(datafile):
    with h5py.File(datafile, "r") as f:
        cfc_embeddings_test = np.array(f["cfc_test"])
        tilde_z = cfc_embeddings_test[:, 1]
        z = cfc_embeddings_test[:, 0]
    return tilde_z, z


def load_oodprobe_paired_samples(
    data_dir, num_samples=1000, train_split=0.9, seed=42
):
    """
    Load CSV files from oodprobe directory and create paired samples with different targets.
    Returns same format as truthful_qa for compatibility with store_embeddings.py.

    Args:
        data_dir: Path to directory containing oodprobe CSV files
        num_samples: Number of paired samples to create per file
        train_split: Fraction of samples for training (default 0.9)
        seed: Random seed for reproducibility

    Returns:
        tuple: (cfc_train_tuples, cfc_test_tuples) where each is a list of [prompt1, prompt2]
    """
    random.seed(seed)
    np.random.seed(seed)

    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    all_pairs = []

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")

        # Load CSV file
        df = pd.read_csv(csv_file)

        # Ensure we have the required columns
        if not all(col in df.columns for col in ["prompt", "target"]):
            print(
                f"Skipping {csv_file.name}: missing 'prompt' or 'target' columns"
            )
            continue

        # Group by target value
        target_groups = df.groupby("target")
        target_values = list(target_groups.groups.keys())

        if len(target_values) < 2:
            print(f"Skipping {csv_file.name}: only one target value found")
            continue

        # Create paired samples with different targets
        for _ in range(num_samples):
            # Randomly select two different target values
            target1, target2 = random.sample(target_values, 2)

            # Get random samples from each target group
            group1 = target_groups.get_group(target1)
            group2 = target_groups.get_group(target2)

            prompt1 = random.choice(group1["prompt"].tolist())
            prompt2 = random.choice(group2["prompt"].tolist())

            # Store as [prompt1, prompt2] matching truthful_qa format
            all_pairs.append([prompt1, prompt2])

    # Shuffle all pairs to mix data from different files
    random.shuffle(all_pairs)

    # Split into train/test
    split_idx = int(len(all_pairs) * train_split)
    cfc_train_tuples = all_pairs[:split_idx]
    cfc_test_tuples = all_pairs[split_idx:]

    print(
        f"Total created: {len(cfc_train_tuples)} train pairs and {len(cfc_test_tuples)} test pairs"
    )

    return cfc_train_tuples, cfc_test_tuples


def load_oodprobe_texts_from_zip(
    zip_path=None,
    max_texts_per_file=None,
    seed=42,
):
    if zip_path is None:
        from data.registry import get_data_root
        zip_path = os.path.join(
            os.path.dirname(get_data_root()), "OODPROBE_DATA", "cleaned_data.zip"
        )
    """
    Load text data from a zipped folder containing cleaned_data.
    Returns individual texts (not pairs) for embedding generation.

    The function expects the zip to contain CSV/JSON/JSONL files with text data.
    For CSV files, it looks for 'prompt', 'text', or 'sentence' columns.
    For JSON/JSONL files, it looks for 'prompt', 'text', or 'sentence' fields.

    Args:
        zip_path: Path to the zip file containing cleaned_data
        max_texts_per_file: Maximum number of texts to load per file (None for all)
        seed: Random seed for reproducible sampling

    Returns:
        list: List of text strings for embedding generation
    """
    import zipfile
    import tempfile
    import io

    random.seed(seed)
    np.random.seed(seed)

    all_texts = []

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    print(f"Loading texts from {zip_path}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # List all files in the zip
        file_list = zf.namelist()
        print(f"Found {len(file_list)} items in zip")

        for file_name in file_list:
            # Skip directories
            if file_name.endswith("/"):
                continue

            # Skip hidden files
            if os.path.basename(file_name).startswith("."):
                continue

            file_texts = []

            try:
                if file_name.endswith(".csv"):
                    # Read CSV file
                    with zf.open(file_name) as f:
                        df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))

                        # Look for text column
                        text_col = None
                        for col_name in [
                            "prompt",
                            "text",
                            "sentence",
                            "content",
                        ]:
                            if col_name in df.columns:
                                text_col = col_name
                                break

                        if text_col is not None:
                            file_texts = (
                                df[text_col].dropna().astype(str).tolist()
                            )
                            print(
                                f"  Loaded {len(file_texts)} texts from {os.path.basename(file_name)} (column: {text_col})"
                            )
                        else:
                            print(
                                f"  Skipping {os.path.basename(file_name)}: no recognized text column"
                            )

                elif file_name.endswith(".jsonl"):
                    # Read JSONL file
                    with zf.open(file_name) as f:
                        content = io.TextIOWrapper(f, encoding="utf-8")
                        for line in content:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    # Look for text field
                                    for field in [
                                        "prompt",
                                        "text",
                                        "sentence",
                                        "content",
                                    ]:
                                        if field in data and data[field]:
                                            file_texts.append(str(data[field]))
                                            break
                                except json.JSONDecodeError:
                                    continue
                    if file_texts:
                        print(
                            f"  Loaded {len(file_texts)} texts from {os.path.basename(file_name)}"
                        )

                elif file_name.endswith(".json"):
                    # Read JSON file
                    with zf.open(file_name) as f:
                        content = io.TextIOWrapper(f, encoding="utf-8").read()
                        data = json.loads(content)

                        # Handle list of dicts
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    for field in [
                                        "prompt",
                                        "text",
                                        "sentence",
                                        "content",
                                    ]:
                                        if field in item and item[field]:
                                            file_texts.append(str(item[field]))
                                            break
                                elif isinstance(item, str):
                                    file_texts.append(item)
                        # Handle dict with text field
                        elif isinstance(data, dict):
                            for field in [
                                "prompt",
                                "text",
                                "sentence",
                                "content",
                            ]:
                                if field in data:
                                    if isinstance(data[field], list):
                                        file_texts.extend(
                                            [str(t) for t in data[field] if t]
                                        )
                                    else:
                                        file_texts.append(str(data[field]))
                                    break

                    if file_texts:
                        print(
                            f"  Loaded {len(file_texts)} texts from {os.path.basename(file_name)}"
                        )

            except Exception as e:
                print(f"  Error processing {os.path.basename(file_name)}: {e}")
                continue

            # Sample if max_texts_per_file is specified
            if (
                max_texts_per_file is not None
                and len(file_texts) > max_texts_per_file
            ):
                file_texts = random.sample(file_texts, max_texts_per_file)

            all_texts.extend(file_texts)

    # Remove duplicates while preserving order
    seen = set()
    unique_texts = []
    for text in all_texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)

    print(f"Total unique texts loaded: {len(unique_texts)}")
    return unique_texts


def create_random_text_pairs(texts, num_pairs=50000, seed=42):
    """
    Create random pairs from a list of texts.

    Args:
        texts: List of text strings
        num_pairs: Number of pairs to create
        seed: Random seed for reproducibility

    Returns:
        list: List of [text1, text2] pairs
    """
    random.seed(seed)

    if len(texts) < 2:
        raise ValueError("Need at least 2 texts to create pairs")

    # Limit num_pairs to avoid excessive computation
    max_possible_pairs = len(texts) * (len(texts) - 1) // 2
    num_pairs = min(num_pairs, max_possible_pairs)

    pairs = []
    for _ in range(num_pairs):
        # Randomly select two different indices
        i, j = random.sample(range(len(texts)), 2)
        pairs.append([texts[i], texts[j]])

    print(
        f"Created {len(pairs)} random pairs from {len(texts)} texts using seed={seed}"
    )
    return pairs


def save_oodprobe_pairs(
    data_dir, output_file, num_samples=1000, train_split=0.9, seed=42
):
    """
    Create and save oodprobe paired samples to a JSON file for reproducibility.

    Args:
        data_dir: Path to directory containing oodprobe CSV files
        output_file: Path to save the paired samples JSON file
        num_samples: Number of paired samples to create per file
        train_split: Fraction of samples for training
        seed: Random seed for reproducibility
    """
    cfc_train_tuples, cfc_test_tuples = load_oodprobe_paired_samples(
        data_dir, num_samples, train_split, seed
    )

    # Save to JSON file
    output_data = {
        "train_pairs": cfc_train_tuples,
        "test_pairs": cfc_test_tuples,
        "metadata": {
            "num_samples_per_file": num_samples,
            "train_split": train_split,
            "seed": seed,
            "total_train": len(cfc_train_tuples),
            "total_test": len(cfc_test_tuples),
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved paired samples to {output_file}")
    return output_file


def load_biasinbios(
    split=0.9,
    num_samples=None,
    data_seed=42,
    *,
    return_labels=False,
    return_metadata=False,
):
    """Sample arbitrary biography pairs from official train/test splits.

    Each row is uniform over all ordered pairs of distinct biographies in its
    source split. Gender and profession never influence sampling. They are
    retained only as factor deltas for evaluation.
    """
    from data.bias_in_bios import (
        assert_disjoint_endpoint_texts,
        audit_pair_sample,
        build_pair_rows,
        load_official_splits,
        prepare_disjoint_endpoint_pools,
        sample_uniform_ordered_pairs,
    )

    if not 0.0 < split < 1.0:
        raise ValueError(f"split must be strictly between 0 and 1, got {split}")

    print("Loading Bias-in-Bios official splits...")
    dataset, source_provenance = load_official_splits()
    (
        train_pool,
        test_pool,
        train_source_indices,
        test_source_indices,
        endpoint_pool_audit,
    ) = prepare_disjoint_endpoint_pools(dataset["train"], dataset["test"])
    total_pairs = (
        min(len(train_pool), len(test_pool))
        if num_samples is None
        else int(num_samples)
    )
    if total_pairs < 2:
        raise ValueError("Bias-in-Bios requires at least two requested pairs")
    train_count = min(max(int(total_pairs * split), 1), total_pairs - 1)
    test_count = total_pairs - train_count

    train_indices = sample_uniform_ordered_pairs(
        len(train_pool), train_count, seed=data_seed
    )
    test_indices = sample_uniform_ordered_pairs(
        len(test_pool), test_count, seed=data_seed + 1
    )
    train_rows = build_pair_rows(
        train_pool,
        train_indices,
        split_name="train",
        source_row_indices=train_source_indices,
    )
    test_rows = build_pair_rows(
        test_pool,
        test_indices,
        split_name="test",
        source_row_indices=test_source_indices,
    )
    train_pairs, train_factors, train_metadata = train_rows
    test_pairs, test_factors, test_metadata = test_rows
    for rows in (train_metadata, test_metadata):
        for row in rows:
            row["source_dataset"] = "LabHC/bias_in_bios"
    assert_disjoint_endpoint_texts(train_metadata, test_metadata)

    train_audit = audit_pair_sample(train_factors, train_metadata)
    test_audit = audit_pair_sample(test_factors, test_metadata)
    print(
        "Created uniform arbitrary pairs: "
        f"train={train_count} from {len(train_pool)} biographies, "
        f"test={test_count} from {len(test_pool)} biographies"
    )
    print(f"Train gender delta counts: {train_audit['gender_delta_counts']}")
    print(f"Test gender delta counts: {test_audit['gender_delta_counts']}")
    print(
        "Eligible endpoint pools: "
        f"train={endpoint_pool_audit['eligible_train_rows']}, "
        f"test={endpoint_pool_audit['eligible_test_rows']} "
        "after exact-text de-duplication"
    )
    print(f"Bias-in-Bios source loader: {source_provenance['loader']}")

    if return_metadata:
        return (
            train_pairs,
            test_pairs,
            train_factors,
            test_factors,
            train_metadata,
            test_metadata,
        )
    if return_labels:
        return train_pairs, test_pairs, train_factors, test_factors
    return train_pairs, test_pairs


def load_categorical_contrastive(split=0.9, num_samples=None, data_seed=42):
    """
    Load categorical contrastive dataset with multi-label concept changes.

    Args:
        split: Train/test split ratio (default: 0.9)
        num_samples: Maximum number of samples to use (default: None for all)
        data_seed: Random seed for reproducible sampling

    Returns:
        tuple: (cfc_train_tuples, cfc_test_tuples, cfc_train_labels, cfc_test_labels)
               where labels are [color_changed, shape_changed, object_changed]
    """
    import json

    # Set random seed for reproducibility
    random.seed(data_seed)
    np.random.seed(data_seed)

    print("Loading categorical contrastive dataset...")

    # Load the train data (contains most samples)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(
        script_dir,
        "..",
        "data",
        "categorical",
        "train.json",
    )

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Categorical dataset not found at {train_file}. "
            "Please run 'python data/categorical.py' to generate it first."
        )

    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} samples in categorical dataset")

    # Sample if requested
    if num_samples is not None and len(data) > num_samples:
        data = random.sample(data, num_samples)
        print(f"Sampled {len(data)} samples")

    # Extract tuples and labels
    cfc_tuples = []
    labels = []

    for sample in data:
        # Create contrastive pair
        cfc_tuples.append([sample["prompt1"], sample["prompt2"]])

        # Convert labels to dict format for consistency with other datasets
        label_dict = {
            "color_changed": bool(sample["labels"][0]),
            "shape_changed": bool(sample["labels"][1]),
            "object_changed": bool(sample["labels"][2]),
        }
        labels.append(label_dict)

    print(f"Created {len(cfc_tuples)} contrastive pairs with 3-concept labels")

    # Split into train/test
    split_idx = int(len(cfc_tuples) * split)
    cfc_train_tuples = cfc_tuples[:split_idx]
    cfc_test_tuples = cfc_tuples[split_idx:]
    cfc_train_labels = labels[:split_idx]
    cfc_test_labels = labels[split_idx:]

    print(
        f"Split: {len(cfc_train_tuples)} train pairs, {len(cfc_test_tuples)} test pairs"
    )

    return cfc_train_tuples, cfc_test_tuples, cfc_train_labels, cfc_test_labels
