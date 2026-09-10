"""
Synthetic manifold benchmark for validating geometry metrics.

Generates additive mixtures of low-dimensional manifolds embedded in
high-dimensional ambient space via random orthonormal maps, producing
contrastive pairs compatible with SSAE training.

Manifold types follow the benchmark design of Bhalla et al. (2026):
circle, sphere, torus, Moebius strip, Swiss roll, helix, flat disk,
line segment.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Manifold parameterizations
# ---------------------------------------------------------------------------
# Each sampler: (n, rng) -> (points (n, d_ext), params (n, d_int))

def _sample_circle(n, rng):
    theta = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return pts, theta[:, None]


def _sample_sphere(n, rng):
    theta = rng.uniform(0, np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=1)
    return pts, np.stack([theta, phi], axis=1)


def _sample_torus(n, rng, R=2.0, r=1.0):
    t1 = rng.uniform(0, 2 * np.pi, n)
    t2 = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([
        (R + r * np.cos(t2)) * np.cos(t1),
        (R + r * np.cos(t2)) * np.sin(t1),
        r * np.sin(t2),
    ], axis=1)
    return pts, np.stack([t1, t2], axis=1)


def _sample_mobius(n, rng):
    theta = rng.uniform(0, 2 * np.pi, n)
    t = rng.uniform(-0.5, 0.5, n)
    pts = np.stack([
        (1 + t * np.cos(theta / 2)) * np.cos(theta),
        (1 + t * np.cos(theta / 2)) * np.sin(theta),
        t * np.sin(theta / 2),
    ], axis=1)
    return pts, np.stack([theta, t], axis=1)


def _sample_swiss_roll(n, rng):
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    h = rng.uniform(0, 1, n)
    pts = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1)
    return pts, np.stack([t, h], axis=1)


def _sample_helix(n, rng):
    t = rng.uniform(0, 4 * np.pi, n)
    pts = np.stack([np.cos(t), np.sin(t), t / (2 * np.pi)], axis=1)
    return pts, t[:, None]


def _sample_disk(n, rng):
    r = np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    return pts, np.stack([r, theta], axis=1)


def _sample_line(n, rng):
    t = rng.uniform(0, 1, n)
    return t[:, None], t[:, None]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManifoldType:
    name: str
    intrinsic_dim: int
    extrinsic_dim: int
    sampler: Callable


MANIFOLD_TYPES = {
    "circle":     ManifoldType("circle",     1, 2, _sample_circle),
    "sphere":     ManifoldType("sphere",     2, 3, _sample_sphere),
    "torus":      ManifoldType("torus",      2, 3, _sample_torus),
    "mobius":     ManifoldType("mobius",      2, 3, _sample_mobius),
    "swiss_roll": ManifoldType("swiss_roll", 2, 3, _sample_swiss_roll),
    "helix":      ManifoldType("helix",      1, 3, _sample_helix),
    "disk":       ManifoldType("disk",       2, 2, _sample_disk),
    "line":       ManifoldType("line",       1, 1, _sample_line),
}

PRESETS = {
    "minimal": ["line", "line"],
    "linear": ["line", "disk", "line"],
    "mixed": ["line", "circle"],
    "curved": ["circle", "helix", "torus"],
    "full": list(MANIFOLD_TYPES.keys()),
}


# ---------------------------------------------------------------------------
# Embedding and pair generation
# ---------------------------------------------------------------------------

def _random_orthonormal(D, d, rng):
    A = rng.randn(D, d)
    Q, _ = np.linalg.qr(A)
    return Q[:, :d]


def generate_benchmark(
    manifold_types: List[str],
    ambient_dim: int = 128,
    n_pairs_per_concept: int = 2000,
    n_test_per_concept: int = 500,
    noise_std: float = 0.0,
    seed: int = 42,
) -> dict:
    """Generate a synthetic manifold benchmark.

    Each manifold instance becomes one concept. For each concept, generates
    contrastive pairs where exactly that concept's manifold parameter changes
    while all others are held fixed.

    The additive model is x = sum_k V_k gamma_k(theta_k), so
    delta_x = V_k (gamma_k(theta') - gamma_k(theta)) for the changing
    concept k. For linear manifolds (line, disk), this is exactly the
    SSAE model. For curved manifolds, it captures chord approximations.

    Returns:
        Dict with train_pairs, test_pairs, concept labels, ground truth
        embedding matrices, and manifold specs.
    """
    rng = np.random.RandomState(seed)
    K = len(manifold_types)

    specs = [MANIFOLD_TYPES[t] for t in manifold_types]
    V = [_random_orthonormal(ambient_dim, s.extrinsic_dim, rng) for s in specs]

    def _make_pairs(n_per_concept):
        all_z, all_zt, all_labels = [], [], []

        for ci in range(K):
            base_pts, _ = specs[ci].sampler(n_per_concept, rng)
            target_pts, _ = specs[ci].sampler(n_per_concept, rng)

            fixed = {}
            for j in range(K):
                if j != ci:
                    pts, _ = specs[j].sampler(n_per_concept, rng)
                    fixed[j] = pts

            z = np.zeros((n_per_concept, ambient_dim))
            zt = np.zeros((n_per_concept, ambient_dim))

            for j in range(K):
                if j == ci:
                    z += base_pts @ V[j].T
                    zt += target_pts @ V[j].T
                else:
                    contrib = fixed[j] @ V[j].T
                    z += contrib
                    zt += contrib

            if noise_std > 0:
                z += rng.randn(*z.shape) * noise_std
                zt += rng.randn(*zt.shape) * noise_std

            all_z.append(z)
            all_zt.append(zt)
            all_labels.append(np.full(n_per_concept, ci))

        z = np.concatenate(all_z)
        zt = np.concatenate(all_zt)
        labels = np.concatenate(all_labels)

        perm = rng.permutation(len(z))
        pairs = np.stack([z[perm], zt[perm]], axis=1)
        return pairs.astype(np.float32), labels[perm].astype(np.int32)

    train_pairs, train_labels = _make_pairs(n_pairs_per_concept)
    test_pairs, test_labels = _make_pairs(n_test_per_concept)

    return {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "train_concept_labels": train_labels,
        "test_concept_labels": test_labels,
        "manifold_specs": [
            {"type": t, "intrinsic_dim": specs[i].intrinsic_dim,
             "extrinsic_dim": specs[i].extrinsic_dim}
            for i, t in enumerate(manifold_types)
        ],
        "embedding_matrices": [v.astype(np.float32) for v in V],
        "config": {
            "ambient_dim": ambient_dim,
            "n_concepts": K,
            "manifold_types": manifold_types,
            "n_train_per_concept": n_pairs_per_concept,
            "n_test_per_concept": n_test_per_concept,
            "noise_std": noise_std,
            "seed": seed,
        },
    }


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

def save_benchmark(data: dict, path: Path) -> None:
    """Save benchmark to HDF5 in SimpleCPUData / load_embeddings format.

    Keys:
        cfc_train: (N_train, 2, D) for SimpleCPUData
        cfc_test: (N_test, 2, D) for load_embeddings
        cfc_test_labels: per-pair concept label strings
        test_concept_labels: per-pair integer concept indices
        gt_embedding_i: ground truth orthonormal map for concept i
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("cfc_train", data=data["train_pairs"])
        f.create_dataset("cfc_test", data=data["test_pairs"])

        f.create_dataset("train_concept_labels", data=data["train_concept_labels"])
        f.create_dataset("test_concept_labels", data=data["test_concept_labels"])

        str_labels = [f"concept_{i}" for i in data["test_concept_labels"]]
        f.create_dataset("cfc_test_labels", data=np.array(str_labels, dtype="S"))

        for i, V in enumerate(data["embedding_matrices"]):
            f.create_dataset(f"gt_embedding_{i}", data=V)

        f.attrs["manifold_specs"] = json.dumps(data["manifold_specs"])
        for k, v in data["config"].items():
            if isinstance(v, list):
                f.attrs[k] = json.dumps(v)
            else:
                f.attrs[k] = v


def load_benchmark(path: Path) -> dict:
    """Load benchmark from HDF5, including ground truth."""
    with h5py.File(path, "r") as f:
        specs = json.loads(f.attrs["manifold_specs"])
        n_concepts = f.attrs["n_concepts"]

        V = [f[f"gt_embedding_{i}"][:] for i in range(n_concepts)]

        return {
            "train_pairs": f["cfc_train"][:],
            "test_pairs": f["cfc_test"][:],
            "train_concept_labels": f["train_concept_labels"][:],
            "test_concept_labels": f["test_concept_labels"][:],
            "manifold_specs": specs,
            "embedding_matrices": V,
            "config": {
                "ambient_dim": int(f.attrs["ambient_dim"]),
                "n_concepts": int(n_concepts),
                "manifold_types": json.loads(f.attrs["manifold_types"]),
                "seed": int(f.attrs["seed"]),
            },
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic manifold benchmark data"
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS.keys()),
        help="Named preset (overrides --manifolds)"
    )
    parser.add_argument(
        "--manifolds", nargs="+", choices=list(MANIFOLD_TYPES.keys()),
        help="Manifold types for each concept"
    )
    parser.add_argument("--ambient-dim", type=int, default=128)
    parser.add_argument("--n-train", type=int, default=2000,
                        help="Training pairs per concept")
    parser.add_argument("--n-test", type=int, default=500,
                        help="Test pairs per concept")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True,
                        help="Output HDF5 path")

    args = parser.parse_args()

    if args.preset:
        manifolds = PRESETS[args.preset]
    elif args.manifolds:
        manifolds = args.manifolds
    else:
        parser.error("Provide --preset or --manifolds")

    print(f"Generating benchmark: {manifolds}")
    print(f"  ambient_dim={args.ambient_dim}, "
          f"n_train={args.n_train}/concept, n_test={args.n_test}/concept")

    data = generate_benchmark(
        manifold_types=manifolds,
        ambient_dim=args.ambient_dim,
        n_pairs_per_concept=args.n_train,
        n_test_per_concept=args.n_test,
        noise_std=args.noise,
        seed=args.seed,
    )

    save_benchmark(data, Path(args.out))
    print(f"Saved to {args.out}")
    print(f"  train: {data['train_pairs'].shape}")
    print(f"  test:  {data['test_pairs'].shape}")

    for i, spec in enumerate(data["manifold_specs"]):
        V = data["embedding_matrices"][i]
        print(f"  concept {i}: {spec['type']} "
              f"(d_int={spec['intrinsic_dim']}, d_ext={spec['extrinsic_dim']}, "
              f"V shape={V.shape})")
