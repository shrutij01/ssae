import random
import json
import os
from datetime import datetime
from base import CATEGORICAL_CODEBOOK


def sample_excluding_elements(items_list, exclude):
    """Sample from list excluding specific elements."""
    filtered_list = [item for item in items_list if item not in exclude]

    if not filtered_list:
        raise ValueError("No elements left to sample after excluding.")

    return random.choice(filtered_list)


def generate_categorical_prompt(shape, color, obj):
    """Generate a natural language prompt describing the categorical object."""
    templates = [
        f"A {color} {shape} {obj}",
        f"The {color} {shape} {obj} is here",
        f"I see a {color} {shape} {obj}",
        f"There is a {color} {shape} {obj}",
        f"Look at this {color} {shape} {obj}",
    ]
    return random.choice(templates)


def generate_contrastive_pair_with_labels(target_changes=None):
    """
    Generate a contrastive pair with specified changes.

    Args:
        target_changes: List of changes to make (e.g., ["color", "shape"])
                       If None, randomly selects 1-3 changes

    Returns:
        tuple: (prompt1, prompt2, labels) where labels = [color_changed, shape_changed, object_changed]
    """
    # Initialize original attributes
    original_shape = random.choice(CATEGORICAL_CODEBOOK["shape"])
    original_color = random.choice(CATEGORICAL_CODEBOOK["color"])
    original_object = random.choice(CATEGORICAL_CODEBOOK["object"])

    # Start with same attributes for the contrast
    new_shape = original_shape
    new_color = original_color
    new_object = original_object

    # Initialize change labels [color_changed, shape_changed, object_changed]
    labels = [0, 0, 0]

    # Determine what to change
    if target_changes is None:
        # Randomly select 1-3 attributes to change
        num_changes = random.randint(1, 3)
        changes_to_make = random.sample(["color", "shape", "object"], num_changes)
    else:
        changes_to_make = target_changes

    # Apply changes and set labels
    for change in changes_to_make:
        if change == "color":
            new_color = sample_excluding_elements(
                CATEGORICAL_CODEBOOK["color"], [original_color]
            )
            labels[0] = 1  # color changed
        elif change == "shape":
            new_shape = sample_excluding_elements(
                CATEGORICAL_CODEBOOK["shape"], [original_shape]
            )
            labels[1] = 1  # shape changed
        elif change == "object":
            new_object = sample_excluding_elements(
                CATEGORICAL_CODEBOOK["object"], [original_object]
            )
            labels[2] = 1  # object changed

    # Generate natural language prompts
    prompt1 = generate_categorical_prompt(original_shape, original_color, original_object)
    prompt2 = generate_categorical_prompt(new_shape, new_color, new_object)

    return prompt1, prompt2, labels


def generate_balanced_dataset(size_per_combination=1000, include_single_changes=True):
    """
    Generate a balanced dataset with all combinations of changes.

    Args:
        size_per_combination: Number of samples per change combination
        include_single_changes: Whether to include single-attribute changes

    Returns:
        list: List of {"prompt1": str, "prompt2": str, "labels": [int, int, int]} dicts
    """
    dataset = []

    # Define all possible change combinations
    change_combinations = []

    if include_single_changes:
        # Single changes
        change_combinations.extend([
            ["color"],
            ["shape"],
            ["object"]
        ])

    # Dual changes
    change_combinations.extend([
        ["color", "shape"],
        ["color", "object"],
        ["shape", "object"]
    ])

    # Triple change
    change_combinations.append(["color", "shape", "object"])

    print(f"Generating {len(change_combinations)} change combinations:")
    for combo in change_combinations:
        print(f"  - {combo}")

    # Generate samples for each combination
    for combination in change_combinations:
        print(f"Generating {size_per_combination} samples for {combination}...")

        for _ in range(size_per_combination):
            prompt1, prompt2, labels = generate_contrastive_pair_with_labels(combination)

            dataset.append({
                "prompt1": prompt1,
                "prompt2": prompt2,
                "labels": labels,
                "change_types": combination  # For debugging/analysis
            })

    # Shuffle the dataset
    random.shuffle(dataset)

    print(f"Generated {len(dataset)} total samples")
    return dataset


def split_dataset(dataset, train_split=0.8, val_split=0.1):
    """Split dataset into train/val/test sets."""
    random.shuffle(dataset)

    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)

    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]

    return train_data, val_data, test_data


def save_dataset_to_json(dataset_splits, output_dir, dataset_name=""):
    """Save dataset splits to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    train_data, val_data, test_data = dataset_splits

    # Save each split
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    for split_name, split_data in splits.items():
        filename = f"{dataset_name + '_' if dataset_name else ''}{split_name}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(split_data)} {split_name} samples to {filepath}")

    # Save dataset statistics
    stats = {
        "dataset_name": dataset_name,
        "created_at": datetime.now().isoformat(),
        "total_samples": len(train_data) + len(val_data) + len(test_data),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "attributes": list(CATEGORICAL_CODEBOOK.keys()),
        "codebook": CATEGORICAL_CODEBOOK,
        "label_format": "[color_changed, shape_changed, object_changed]"
    }

    stats_file = os.path.join(output_dir, f"{dataset_name + '_' if dataset_name else ''}stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset statistics to {stats_file}")


def analyze_dataset_balance(dataset):
    """Analyze the balance of change types in the dataset."""
    from collections import Counter

    # Count label combinations
    label_combinations = Counter()
    change_type_counts = Counter()

    for sample in dataset:
        labels = tuple(sample["labels"])
        label_combinations[labels] += 1

        change_types = tuple(sorted(sample["change_types"]))
        change_type_counts[change_types] += 1

    print("\nDataset Balance Analysis:")
    print("=" * 40)

    print("\nLabel combinations [color, shape, object]:")
    for labels, count in sorted(label_combinations.items()):
        percentage = (count / len(dataset)) * 100
        print(f"  {labels}: {count} samples ({percentage:.1f}%)")

    print("\nChange type combinations:")
    for change_types, count in sorted(change_type_counts.items()):
        percentage = (count / len(dataset)) * 100
        print(f"  {change_types}: {count} samples ({percentage:.1f}%)")


def main():
    """Generate the categorical contrastive dataset."""
    # Set random seed for reproducibility
    random.seed(42)

    print("Generating Categorical Contrastive Dataset")
    print("=" * 45)

    # Generate balanced dataset
    dataset = generate_balanced_dataset(
        size_per_combination=1500,  # 1500 samples per combination
        include_single_changes=True
    )

    # Analyze dataset balance
    analyze_dataset_balance(dataset)

    # Split dataset
    train_data, val_data, test_data = split_dataset(dataset, train_split=0.7, val_split=0.15)

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # Save dataset
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "categorical")
    save_dataset_to_json((train_data, val_data, test_data), output_dir)

    print(f"\n✅ Dataset generation complete!")
    print(f"📁 Files saved to: {output_dir}")


if __name__ == "__main__":
    main()