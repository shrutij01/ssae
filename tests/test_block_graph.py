import numpy as np
import pytest
import torch

from experiments.legacy.block_graph import (
    AffinityConfig,
    PartitionCandidate,
    block_dimension_statistics,
    build_affinity_graph,
    evaluate_partition,
    graph_partition_candidates,
    group_contribution_energies,
    paired_differences,
    partition_description_bits,
    random_partition_like,
)


def test_paired_differences_uses_counterfactual_minus_base():
    embeddings = torch.tensor(
        [
            [1.0, 2.0],
            [4.0, 8.0],
            [-1.0, 3.0],
            [2.0, 1.0],
        ]
    )

    differences = paired_differences(embeddings)

    torch.testing.assert_close(
        differences,
        torch.tensor([[3.0, 6.0], [3.0, -2.0]]),
    )


def test_paired_differences_rejects_an_unpaired_row():
    with pytest.raises(ValueError, match="even number"):
        paired_differences(torch.zeros(3, 2))


def test_affinity_is_built_from_absolute_signed_codes():
    base = torch.linspace(-2.0, 2.0, 101)
    independent = torch.sin(torch.linspace(0.0, 8.0, 101))
    codes = torch.stack([base, -base, independent], dim=1)

    graph = build_affinity_graph(
        codes,
        config=AffinityConfig(
            amplitude_weight=1.0,
            support_weight=0.0,
            decoder_weight=0.0,
            knn=2,
            max_features=None,
        ),
    )

    assert graph.amplitude_affinity[0, 1] == pytest.approx(1.0)
    assert graph.affinity[0, 1] == pytest.approx(1.0)


def test_graph_candidates_have_variable_sizes_and_respect_the_cap():
    rng = np.random.default_rng(0)
    n = 200
    first = rng.normal(size=n)
    second = first + 0.01 * rng.normal(size=n)
    third = rng.normal(size=n)
    fourth = third + 0.01 * rng.normal(size=n)
    codes = torch.tensor(
        np.stack([first, second, third, fourth], axis=1),
        dtype=torch.float32,
    )
    graph = build_affinity_graph(
        codes,
        config=AffinityConfig(
            amplitude_weight=1.0,
            support_weight=0.0,
            knn=2,
            max_features=None,
        ),
    )

    candidates = graph_partition_candidates(
        graph,
        total_features=4,
        edge_quantiles=[0.5],
        max_block_size=2,
    )

    assert candidates[0].groups == ((0,), (1,), (2,), (3,))
    assert all(
        max(len(group) for group in candidate.groups) <= 2
        for candidate in candidates
    )
    assert any(any(len(group) == 2 for group in candidate.groups) for candidate in candidates)


def test_contribution_energy_accounts_for_decoder_cancellation():
    codes = torch.tensor([[1.0, -1.0], [1.0, 1.0]])
    decoder = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    energy = group_contribution_energies(codes, decoder, [(0, 1)])

    torch.testing.assert_close(energy[:, 0], torch.tensor([0.0, 2.0]))


def test_block_dimension_statistics_recovers_an_isotropic_plane():
    theta = torch.linspace(0.0, 2.0 * torch.pi, 512)
    codes = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    decoder = torch.eye(2)

    result = block_dimension_statistics(codes, decoder, [(0, 1)])[0]

    assert result["membership_size"] == 2
    assert result["linear_rank"] == 2
    assert result["stable_rank"] == pytest.approx(2.0, rel=2e-2)
    assert result["participation_ratio"] == pytest.approx(2.0, rel=2e-2)
    assert result["effective_rank"] == pytest.approx(2.0, rel=2e-2)
    assert result["pca_dimension"] == 2


def test_partition_evaluation_uses_top_contribution_energy_and_reconstructs():
    codes = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 3.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    decoder = torch.eye(2)
    targets = codes @ decoder.T
    candidate = PartitionCandidate(
        name="singletons",
        groups=((0,), (1,)),
        source="singleton",
    )

    result = evaluate_partition(
        candidate,
        codes=codes,
        decoder=decoder,
        decoder_bias=torch.zeros(2),
        targets=targets,
        active_blocks=1,
        distortion_fraction=0.2,
    )

    assert result["active_blocks"] == 1
    assert result["r2"] == pytest.approx(1.0)
    assert np.isfinite(result["total_bits"])


def test_random_control_preserves_block_sizes_and_partition_coverage():
    candidate = PartitionCandidate(
        name="candidate",
        groups=((0, 1, 2), (3, 4), (5,)),
        source="graph",
    )

    random_control = random_partition_like(candidate, seed=7)

    assert sorted(map(len, random_control.groups)) == [1, 2, 3]
    assert sorted(i for group in random_control.groups for i in group) == list(range(6))
    assert random_control.source == "random_control"


def test_partition_description_cost_is_zero_at_both_trivial_extremes():
    assert partition_description_bits([(0,), (1,), (2,)], 100) == pytest.approx(0.0)
    assert partition_description_bits([(0, 1, 2)], 100) == pytest.approx(0.0)
    assert partition_description_bits([(0, 1), (2,)], 100) == pytest.approx(
        np.log2(3) / 100
    )
