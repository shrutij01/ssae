import torch as t
import h5py
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, List
import yaml
import os
import glob
from sklearn.metrics import roc_auc_score
import numpy as np


def load_oodprobe_test_data(
    data_dir: str, dataset_name: str
) -> Tuple[List[str], List[int]]:
    """Load test data for a specific oodprobe dataset and binarize labels."""
    csv_file = os.path.join(data_dir, f"{dataset_name}.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    sentences = df["prompt"].tolist()
    raw_labels = df["target"].tolist()

    # Get unique labels to understand the distribution
    unique_labels = list(set(raw_labels))
    print(f"Dataset {dataset_name}: Found unique labels: {unique_labels}")

    # Binarize labels based on their values
    labels = []

    if len(unique_labels) == 2:
        # Binary classification - need to determine which is 0 and which is 1
        if all(isinstance(label, (int, float)) for label in raw_labels):
            # Already numeric (0/1), keep as is
            labels = [int(label) for label in raw_labels]
        else:
            # String labels - sort them and assign 0 to first, 1 to second
            sorted_unique = sorted(unique_labels)
            label_map = {sorted_unique[0]: 0, sorted_unique[1]: 1}
            labels = [label_map[label] for label in raw_labels]
            print(f"Label mapping: {label_map}")
    else:
        raise ValueError(
            f"Expected 2 unique labels, got {len(unique_labels)}: {unique_labels}"
        )

    # Print label distribution
    positive_count = sum(labels)
    total_count = len(labels)
    print(
        f"Dataset {dataset_name}: Label distribution: {positive_count}/{total_count} positive ({positive_count/total_count:.2%})"
    )

    return sentences, labels


def score_identification(acts, labels, lamda=0.1, metric="accuracy"):
    """Evaluate model performance using different metrics."""
    scores = {}
    top_features = {}

    # Convert labels to tensor if needed
    if not isinstance(labels, t.Tensor):
        labels = t.tensor(labels, dtype=t.bool)

    feature_labels = acts.T > lamda  # F x N

    if metric == "accuracy":
        matches = feature_labels == labels
        accuracies = matches.sum(dim=1) / labels.shape[-1]
        accuracy = accuracies.max()
        top_feature = accuracies.argmax()
        scores["binary_classification"] = accuracy
        top_features["binary_classification"] = top_feature

    elif metric == "macrof1":
        # Calculate true positives, false positives, false negatives for each feature
        true_positives = (feature_labels & labels).sum(dim=1).float()  # F
        false_positives = (feature_labels & ~labels).sum(dim=1).float()  # F
        false_negatives = (~feature_labels & labels).sum(dim=1).float()  # F

        # Calculate precision and recall
        precision = true_positives / (
            true_positives + false_positives + 1e-10
        )  # F
        recall = true_positives / (
            true_positives + false_negatives + 1e-10
        )  # F

        # Calculate F1 scores
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)  # F

        # Find the feature with the max F1 score
        top_feature = f1_scores.argmax()
        max_f1 = f1_scores[top_feature]

        scores["binary_classification"] = max_f1
        top_features["binary_classification"] = top_feature

    elif metric == "mcc":
        acts_centered = acts - acts.mean(dim=0, keepdim=True)
        acts_std = acts_centered.norm(dim=0, keepdim=True)

        # Convert boolean labels to float for mean calculation
        labels_float = labels.float()
        labels_centered = labels_float - labels_float.mean()
        labels_std = labels_centered.norm()

        # Correlation computation
        numerator = acts_centered.T @ labels_centered  # F
        denominator = acts_std.squeeze() * labels_std  # F

        mask = denominator != 0  # prevent NaNs
        corr_vector = t.zeros_like(numerator)
        corr_vector[mask] = numerator[mask] / denominator[mask]

        # Get index of maximum correlation
        top_feature = corr_vector.argmax()
        max_corr = corr_vector[top_feature]

        scores["binary_classification"] = max_corr
        top_features["binary_classification"] = top_feature

        return corr_vector, top_features

    elif metric == "auc":
        # Calculate AUC for each feature
        labels_np = labels.cpu().numpy()
        aucs = []

        for i in range(acts.shape[1]):  # For each feature
            feature_acts = acts[:, i].cpu().numpy()
            try:
                auc = roc_auc_score(labels_np, feature_acts)
                aucs.append(auc)
            except ValueError:
                # Handle case where all labels are the same
                aucs.append(0.5)

        aucs = t.tensor(aucs)
        top_feature = aucs.argmax()
        max_auc = aucs[top_feature]

        scores["binary_classification"] = max_auc
        top_features["binary_classification"] = top_feature

    else:
        raise ValueError(f"Unrecognized metric: {metric}")

    return scores, top_features


def load_model(model_path: Path):
    """Load trained SSAE model."""
    from ssae import DictLinearAE

    # Load model config (prioritize cfg.yaml)
    config_files = list(model_path.glob("cfg.yaml")) + list(
        model_path.glob("*.yaml")
    )
    if not config_files:
        raise FileNotFoundError(f"No config file found in {model_path}")

    config_path = config_files[0]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load model weights
    weights_files = list(model_path.glob("weights.pth")) + list(
        model_path.glob("checkpoint.pt")
    )
    if not weights_files:
        raise FileNotFoundError(f"No weights file found in {model_path}")

    weights_path = weights_files[0]
    state_dict = t.load(weights_path, map_location="cpu")

    # Get dimensions from the saved weights or config
    if "encoder.weight" in state_dict:
        rep_dim = state_dict["encoder.weight"].shape[1]  # input dimension
        hid_dim = state_dict["encoder.weight"].shape[0]  # hidden dimension
    else:
        # Fallback to config
        rep_dim = config.get("rep_dim", config.get("n_inputs", 512))
        hid_dim = config.get(
            "hid_dim", config.get("encoding_dim", config.get("n_hidden", 4096))
        )

    # Initialize model with correct parameters. input_norm + fold keep this in
    # step with the SSAE-owned loader (DictLinearAE.from_pretrained) for
    # standardize checkpoints; the flexible file discovery above is retained.
    model = DictLinearAE(
        rep_dim=rep_dim,
        hid=hid_dim,
        norm_type=config.get("norm", config.get("norm_type", "ln")),
        skip_input_norm=config.get("skip_input_norm", False),
        activation=config.get("activation", "none"),
        input_norm=config.get("input_norm", "standardize"),
        soft_threshold_init=config.get("soft_threshold_init", 0.01),
    )

    model.load_state_dict(state_dict)
    model.fold_input_norm_()
    model.eval()

    return model


def get_sentence_embeddings(
    sentences,
    model_name="EleutherAI/pythia-70m-deduped",
    layer=5,
    batch_size=128,
):
    """Extract embeddings for individual sentences using store_embeddings function."""
    from ssae.store_embeddings import extract_embeddings
    import torch
    from transformers import GPTNeoXForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the language model
    if model_name == "EleutherAI/pythia-70m-deduped":
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000",
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        raise NotImplementedError(f"Model {model_name} not supported")

    # Convert individual sentences to fake pairs format for extract_embeddings
    # We'll duplicate each sentence so we get (sentence, sentence) pairs
    fake_contexts = [(sentence, sentence) for sentence in sentences]

    # Extract embeddings using the existing function
    embeddings = extract_embeddings(
        fake_contexts, model, tokenizer, layer, "last_token", batch_size
    )

    # Since we used (sentence, sentence) pairs, the difference will be zero
    # So we just take the first embedding from each pair
    return embeddings[:, 0, :]  # Shape: (N, rep_dim)


def get_activations(
    model: t.nn.Module, embeddings: t.Tensor, batch_size: int = 512
) -> t.Tensor:
    """Get SSAE activations for embeddings."""
    model.eval()
    all_activations = []

    with t.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size]
            # Preserve the trained input norm, code norm, and activation gate.
            # Calling encoder directly would bypass ReLU/softshrink ablations.
            _, activations = model(batch)
            all_activations.append(activations.cpu())

    return t.cat(all_activations, dim=0)


def find_model_dirs_with_seeds(
    base_dir: Path, dataset_name: str, seeds: List[int]
) -> List[Path]:
    """Find model directories for a specific dataset across different seeds."""
    model_dirs = []

    # Multiple pattern options to handle different naming conventions
    for seed in seeds:
        patterns = [
            base_dir / f"{dataset_name}_seed{seed}",  # dataset_name_seed0
            base_dir / f"{dataset_name}_seed_{seed}",  # dataset_name_seed_0
            base_dir / f"{dataset_name}" / f"seed_{seed}",  # dataset_name/seed_0/
            base_dir / f"{dataset_name}" / f"seed{seed}",  # dataset_name/seed0/
        ]

        for pattern in patterns:
            if pattern.exists():
                model_dirs.append(pattern)
                break  # Found one, move to next seed

    print(f"Searched for {dataset_name} models: found {len(model_dirs)} directories")
    return model_dirs


def evaluate_dataset_across_seeds(
    base_model_dir: Path,
    data_dir: Path,
    dataset_name: str,
    seeds: List[int],
    threshold: float = 0.1,
    metrics: List[str] = ["accuracy", "macrof1", "mcc", "auc"],
) -> Dict[str, Any]:
    """Evaluate a specific oodprobe dataset across multiple seeds."""

    # Load test data
    sentences, labels = load_oodprobe_test_data(str(data_dir), dataset_name)
    print(f"Loaded {len(sentences)} test samples for {dataset_name}")

    # Get sentence embeddings (only need to do this once)
    print("Extracting sentence embeddings...")
    embeddings = get_sentence_embeddings(sentences)
    print(f"Embeddings shape: {embeddings.shape}")

    # Convert embeddings to tensor if needed
    if not isinstance(embeddings, t.Tensor):
        embeddings = t.tensor(embeddings, dtype=t.float32)
        print(f"Converted embeddings to tensor: {embeddings.shape}")

    # Find model directories for this dataset
    model_dirs = find_model_dirs_with_seeds(
        base_model_dir, dataset_name, seeds
    )
    print(f"Found {len(model_dirs)} model directories for {dataset_name}")

    if not model_dirs:
        print(f"Warning: No model directories found for {dataset_name}")
        return {}

    # Evaluate each seed
    seed_results = {}
    for model_dir in model_dirs:
        try:
            # Extract seed from directory name
            if "seed_" in str(model_dir):
                seed = int(str(model_dir).split("seed_")[-1])
            else:
                seed = "unknown"

            print(f"Evaluating seed {seed}...")

            # Load model
            model = load_model(model_dir)

            # Get SSAE activations
            activations = get_activations(model, embeddings)
            print(f"Activations shape: {activations.shape}")

            # Evaluate for each metric
            seed_metrics = {}
            for metric in metrics:
                if metric == "mcc":
                    corr_vector, top_features = score_identification(
                        activations, labels, threshold, metric
                    )
                    seed_metrics[metric] = {
                        "score": float(corr_vector.max().item()),
                        "top_feature": int(
                            top_features["binary_classification"]
                        ),
                    }
                else:
                    scores, top_features = score_identification(
                        activations, labels, threshold, metric
                    )
                    seed_metrics[metric] = {
                        "score": float(scores["binary_classification"]),
                        "top_feature": int(
                            top_features["binary_classification"]
                        ),
                    }

            seed_results[f"seed_{seed}"] = seed_metrics

        except Exception as e:
            print(f"Error evaluating seed {seed}: {e}")
            continue

    return seed_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SSAE models on oodprobe datasets across multiple seeds"
    )
    parser.add_argument(
        "model_base_dir",
        type=Path,
        help="Base directory containing trained models organized by dataset/seed",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "oodprobe" / "ood",
        help="Directory containing oodprobe CSV files",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to evaluate (if not provided, evaluates all CSV files)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 5, 7],
        help="Seeds to evaluate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Activation threshold for binarization",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "macrof1", "mcc", "auc"],
        choices=["accuracy", "macrof1", "mcc", "auc"],
        help="Metrics to compute",
    )
    parser.add_argument("--output", type=Path, help="Output file for results")

    args = parser.parse_args()

    # Get list of datasets to evaluate
    if args.datasets:
        datasets = args.datasets
    else:
        # Get all CSV files in data directory
        csv_files = list(args.data_dir.glob("*.csv"))
        datasets = [f.stem for f in csv_files]
        print(f"Found CSV files: {[f.name for f in csv_files]}")

    print(f"Evaluating datasets: {datasets}")
    print(f"Using seeds: {args.seeds}")
    print(f"Using metrics: {args.metrics}")

    # Evaluate each dataset
    all_results = {}
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {dataset}")
        print(f"{'='*60}")

        dataset_results = evaluate_dataset_across_seeds(
            args.model_base_dir,
            args.data_dir,
            dataset,
            args.seeds,
            args.threshold,
            args.metrics,
        )

        if dataset_results:
            all_results[dataset] = dataset_results

            # Print summary for this dataset
            print(f"\nSummary for {dataset}:")
            for metric in args.metrics:
                scores = []
                for seed_key, seed_data in dataset_results.items():
                    if metric in seed_data:
                        scores.append(seed_data[metric]["score"])

                if scores:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    print(
                        f"  {metric}: {mean_score:.4f} ± {std_score:.4f} (n={len(scores)})"
                    )

    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    for metric in args.metrics:
        print(f"\n{metric.upper()}:")
        for dataset, dataset_results in all_results.items():
            scores = []
            for seed_key, seed_data in dataset_results.items():
                if metric in seed_data:
                    scores.append(seed_data[metric]["score"])

            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"  {dataset}: {mean_score:.4f} ± {std_score:.4f}")

    # Save results if output specified
    if args.output:
        # Add summary statistics
        summary_results = {
            "datasets": all_results,
            "summary": {},
        }

        for metric in args.metrics:
            summary_results["summary"][metric] = {}
            for dataset, dataset_results in all_results.items():
                scores = []
                for seed_key, seed_data in dataset_results.items():
                    if metric in seed_data:
                        scores.append(seed_data[metric]["score"])

                if scores:
                    summary_results["summary"][metric][dataset] = {
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                        "count": len(scores),
                        "scores": [float(s) for s in scores],
                    }

        with open(args.output, "w") as f:
            json.dump(summary_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
