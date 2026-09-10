#!/usr/bin/env python3
"""
Generate embeddings for categorical-contrastive dataset using multiple models.

This script generates embeddings for the categorical-contrastive dataset using:
- Llama-3.1-8B-Instruct (layer 31, batch size 4)
- Gemma-2-2b-it (layer 16, batch size 8)
- Pythia-70m-deduped (layer 5, batch size 128)

Each model uses optimized layer and batch size configurations.
"""

import subprocess
import sys
import time
from datetime import datetime
import argparse
import os


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ SUCCESS - Duration: {duration:.1f}s")
        if result.stdout:
            print("Output:")
            print(result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED - Exit code: {e.returncode}")
        print(f"Error: {e.stderr}")
        return False

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT - Command exceeded 2 hours")
        return False

    except KeyboardInterrupt:
        print(f"🛑 INTERRUPTED by user")
        return False


def generate_embeddings_for_model(model_config, args):
    """Generate embeddings for a single model configuration."""
    model_id = model_config["model_id"]
    layer = model_config["layer"]
    batch_size = model_config["batch_size"]
    model_name = model_config["name"]

    print(f"\n🔧 Generating embeddings for {model_name}")
    print(f"   Model: {model_id}")
    print(f"   Layer: {layer}")
    print(f"   Batch size: {batch_size}")
    print(f"   Samples: {args.num_samples}")

    # Build command (script is run from ssae/ directory)
    cmd = [
        "python", "-m", "ssae.store_embeddings",
        "--dataset", "categorical-contrastive",
        "--model_id", model_id,
        "--layer", str(layer),
        "--batch-size", str(batch_size),
        "--pooling-method", "last_token"
    ]

    # Add optional arguments
    if args.num_samples:
        cmd.extend(["--num-samples", str(args.num_samples)])

    if args.split:
        cmd.extend(["--split", str(args.split)])

    if args.quick:
        cmd.append("--quick")

    description = f"Generating {model_name} embeddings"
    return run_command(cmd, description)


def main():
    parser = argparse.ArgumentParser(
        description="Generate categorical-contrastive embeddings for multiple models"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=7000,
        help="Number of samples to process (default: 7000)"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        help="Train/test split ratio (default: 0.9)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick mode (5500 samples)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gemma", "pythia", "all"],
        default=["all"],
        help="Which models to run (default: all - currently gemma and pythia)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have embeddings generated"
    )

    args = parser.parse_args()

    # Model configurations optimized for each architecture
    model_configs = {
        "llama": {
            "name": "Llama-3.1-8B-Instruct",
            "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "layer": 31,  # Final layer (0-indexed, so layer 31 = 32nd layer)
            "batch_size": 4,  # Very conservative for 8B model
            "expected_dim": 4096
        },
        "gemma": {
            "name": "Gemma-2-2b-it",
            "model_id": "google/gemma-2-2b-it",
            "layer": 16,  # Good concept formation layer
            "batch_size": 8,  # Conservative batch size for 2B model
            "expected_dim": 2304
        },
        "pythia": {
            "name": "Pythia-70m-deduped",
            "model_id": "EleutherAI/pythia-70m-deduped",
            "layer": 5,  # Final layer for 6-layer model
            "batch_size": 128,  # Large batch size for small model
            "expected_dim": 512
        }
    }

    # Determine which models to run (skip llama for now)
    if "all" in args.models:
        models_to_run = ["gemma", "pythia"]  # Skip llama due to access issues
    else:
        models_to_run = [m for m in args.models if m != "all" and m != "llama"]

    print("🎯 Categorical Contrastive Embedding Generation")
    print("=" * 60)
    print(f"Dataset: categorical-contrastive")
    print(f"Models to run: {', '.join(models_to_run)}")
    print(f"Samples: {'5500 (quick mode)' if args.quick else args.num_samples}")
    print(f"Split: {args.split}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if dataset exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(
        script_dir, "..", "data", "categorical_dataset", "categorical_contrastive_train.json"
    )

    if not os.path.exists(dataset_file):
        print(f"\n❌ ERROR: Categorical dataset not found at {dataset_file}")
        print("Please run 'python data/categorical.py' first to generate the dataset.")
        sys.exit(1)

    print(f"✅ Dataset found: {dataset_file}")

    # Track results
    results = {}
    successful_runs = 0
    failed_runs = 0

    # Generate embeddings for each model
    for model_key in models_to_run:
        model_config = model_configs[model_key]

        success = generate_embeddings_for_model(model_config, args)
        results[model_key] = success

        if success:
            successful_runs += 1
            print(f"✅ {model_config['name']} completed successfully")
        else:
            failed_runs += 1
            print(f"❌ {model_config['name']} failed")

        # Add delay between models to prevent resource conflicts
        if model_key != models_to_run[-1]:  # Don't delay after the last model
            print("⏳ Waiting 30 seconds before next model...")
            time.sleep(30)

    # Final summary
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total models: {len(models_to_run)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n📁 Expected output files:")
    for model_key in models_to_run:
        if results[model_key]:
            model_config = model_configs[model_key]
            layer = model_config["layer"]
            expected_dim = model_config["expected_dim"]
            model_short = "llama3" if model_key == "llama" else ("gemma2" if model_key == "gemma" else "pythia70m")
            print(f"✅ {model_config['name']}:")
            print(f"   📄 categorical-contrastive_{model_short}_{layer}_last_token.h5")
            print(f"   📄 categorical-contrastive_{model_short}_{layer}_last_token.yaml")
            print(f"   🔢 Expected embedding dim: {expected_dim}")
        else:
            print(f"❌ {model_configs[model_key]['name']}: Failed")

    print(f"\n  Files saved to: $SSAE_DATA_ROOT/categorical-contrastive/")

    # Set exit code based on results
    if failed_runs > 0:
        print(f"\n⚠️  Some models failed. Check logs above for details.")
        sys.exit(1)
    else:
        print(f"\n🎉 All models completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n🛑 Script interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)