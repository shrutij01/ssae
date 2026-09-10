"""
Evaluate Out-of-Distribution (OOD) data using steering vectors.
"""

import sys
import os


import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import utils.data_utils as utils
from loaders import (
    TestDataLoader,
    load_llamascope_checkpoint,
    load_ssae_models,
)


def get_ood_cosine_similarity(steering_vector, ood_data, decoder_bias, device):
    """
    Compute cosine similarity between steering vector and OOD data.
    """
    # Use TestDataLoader for consistent data loading
    loader = TestDataLoader(device=device, verbose=False)

    # Find dataconfig path (assume it's in same directory as ood_data)
    ood_dir = os.path.dirname(os.path.abspath(ood_data))
    dataconfig_path = os.path.join(ood_dir, "dataconfig.yaml")

    tensors, _, _, status = loader.load_test_data(ood_data, dataconfig_path)
    if tensors is None:
        print(f"Failed to load OOD data: {status}")
        return None

    tilde_z_ood, z_ood = tensors
    tilde_z_ood = F.normalize(tilde_z_ood, dim=1)
    z_ood = F.normalize(z_ood, dim=1)
    decoder_bias = decoder_bias.to(z_ood.device)

    # steering vector is already normalized
    tilde_hat_z_ood = (
        z_ood + steering_vector.unsqueeze(0) + decoder_bias.unsqueeze(0)
    )

    cosines = []
    tilde_hat_z_ood = utils.numpify(tilde_hat_z_ood)
    tilde_z_ood = utils.numpify(tilde_z_ood)

    for i in range(tilde_z_ood.shape[0]):
        cosines.append(
            cosine_similarity(
                tilde_z_ood[i].reshape(1, -1),
                tilde_hat_z_ood[i].reshape(1, -1),
            )
        )

    mean_cosine = np.mean(cosines)
    std_cosine = np.std(cosines)
    print(f"OOD Cosine Similarities: {mean_cosine:.4f} ± {std_cosine:.4f}")

    return {"cosines": cosines, "mean": mean_cosine, "std": std_cosine}


def main(args):
    """Evaluate OOD data using steering vectors."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load main test data to get steering vectors
    loader = TestDataLoader(device=device, verbose=args.verbose)
    tensors, dataconfig, test_labels, status = loader.load_test_data(
        args.datafile, args.dataconfig
    )

    if tensors is None:
        print(f"Failed to load test data: {status}")
        return

    tilde_z_test, z_test = tensors
    print(f"Data loaded successfully: {status}")

    # Split data by concept
    concept_test_sets = loader.split_by_label(
        tilde_z=tilde_z_test, z=z_test, labels=test_labels
    )

    if concept_test_sets is None:
        print("Failed to split data by concepts")
        return

    # Load steering vector
    if not os.path.exists(args.steering_vector):
        print(f"Steering vector file not found: {args.steering_vector}")
        return

    steering_vector = torch.load(args.steering_vector, map_location=device)
    print(f"Loaded steering vector from: {args.steering_vector}")

    # Load decoder bias based on model type
    if args.modeltype == "llamascope":
        _, decoder_bias, _, _ = load_llamascope_checkpoint()
        decoder_bias = decoder_bias.to(device)
    elif args.modeltype == "ssae":
        _, decoder_bias_vectors, _, _ = load_ssae_models([args.modeldir])
        decoder_bias = decoder_bias_vectors[0].to(device)
    else:
        raise ValueError("Invalid model type. Choose 'llamascope' or 'ssae'")

    # Get OOD concept ID to verify matching
    ood_dir = os.path.dirname(os.path.abspath(args.ood_data))
    ood_concept_id = utils.load_json(
        os.path.join(ood_dir, "concept_labels_test.json")
    )[0]

    print(f"OOD concept ID: {ood_concept_id}")

    # Evaluate OOD similarity
    results = get_ood_cosine_similarity(
        steering_vector, args.ood_data, decoder_bias, device
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate OOD data with steering vectors"
    )
    parser.add_argument(
        "--datafile", required=True, help="Path to test data file"
    )
    parser.add_argument(
        "--dataconfig", required=True, help="Path to data configuration YAML"
    )
    parser.add_argument(
        "--ood-data", required=True, help="Path to OOD data file"
    )
    parser.add_argument(
        "--steering-vector", required=True, help="Path to steering vector file"
    )
    parser.add_argument(
        "--modeltype",
        required=True,
        choices=["llamascope", "ssae"],
        help="Type of model used",
    )
    parser.add_argument(
        "--modeldir", help="Path to model directory (for SSAE)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.modeltype == "ssae" and not args.modeldir:
        parser.error("--modeldir is required when using SSAE model type")

    main(args)
