"""Acceptance tests for the v4 sparsity & normalisation patch (docs/v4)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ssae.activations import SignedBatchTopKGate, SignedHardGate
from ssae.config import Cfg, hash_cfg
from ssae.constraints import SSAEConstrainedProblem
from ssae.model import DictLinearAE
from ssae.selection import gated_error, pick_beta_elbow, select_r, select_tau
from ssae.sparsity_utils import exact_count, posthoc_gate, subspace_l0, surrogate_count
from ssae.subspace_trainer import export, fit, subspace_l0_batched


def _unit_model(gate="signed_hard", d=12, m=5, tau=0.1, beta=1.5, seed=0):
    torch.manual_seed(seed)
    return DictLinearAE(d, m, "none", activation=gate, input_norm="unit", zero_biases=True,
                        signed_hard_tau=tau, batchtopk_beta=beta)


# ------------------------------------------------------------------ gates
def test_signed_hard_exact_zeros_identity_gradient_and_odd():
    gate = SignedHardGate(0.1)
    h = torch.tensor([[-0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]], requires_grad=True)
    out = gate(h)
    expected = torch.tensor([[-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])
    torch.testing.assert_close(out, expected)
    assert (out == 0).sum() == 5  # exact zeros, including |h| == tau
    (out * torch.arange(7.0)).sum().backward()
    torch.testing.assert_close(h.grad, torch.arange(7.0)[None])  # straight-through identity
    x = torch.randn(64, 9)
    torch.testing.assert_close(gate(-x), -gate(x))


def test_signed_batchtopk_keeps_round_beta_b_and_eval_uses_theta():
    gate = SignedBatchTopKGate(beta=1.5, momentum=0.0)
    gate.train()
    torch.manual_seed(0)
    for B in (8, 10, 33):
        h = torch.randn(B, 7)
        out = gate(h)
        assert int((out != 0).sum()) == round(1.5 * B)
        kept = out[out != 0].abs()
        dropped = h[out == 0].abs()
        assert kept.min() >= dropped.max()
        torch.testing.assert_close(gate(-h), -out)  # odd
    theta = float(gate.theta)
    gate.eval()
    h = torch.randn(50, 7)
    torch.testing.assert_close(gate(h), h * (h.abs() > theta))


def test_signed_batchtopk_eval_requires_training_threshold():
    gate = SignedBatchTopKGate(beta=1.0)
    gate.eval()
    with pytest.raises(RuntimeError):
        gate(torch.randn(4, 3))


def test_signed_batchtopk_theta_is_saved_in_state_dict():
    model = _unit_model("signed_batchtopk")
    model.train()
    model(torch.randn(32, 12))
    state = model.state_dict()
    assert "activation.theta" in state and torch.isfinite(state["activation.theta"])


# ------------------------------------------------------------------ count units
def test_count_is_per_pair_not_fraction_of_width():
    h = torch.tensor([[0.5, 0.0, -0.2, 0.01], [0.0, 0.0, 0.3, -0.06]])
    assert float(exact_count(h, 0.05)) == pytest.approx(2.0)  # (2 + 2) / 2 pairs
    torch.testing.assert_close(surrogate_count(torch.zeros(3, 4), 0.05), torch.tensor(0.0))
    cmp = SSAEConstrainedProblem(torch.device("cpu"), batch=2, hid=4, n_concepts=4, warmup=0,
                                 schedule=1, target=1.5, sparsity_type="count",
                                 budget_units="per_pair", count_threshold=0.05)
    cmp.epoch = 1
    model = DictLinearAE(4, 4, "none", activation="signed_hard", input_norm="unit",
                         zero_biases=True, signed_hard_tau=0.05)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(4))
        model.decoder.weight.copy_(torch.eye(4))
    x = torch.tensor([[3.0, 0.0, 4.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    state = cmp.compute_cmp_state(model, x)
    cs = state.observed_constraints[cmp.sparsity_constraint]
    # unit codes: [0.6, 0, 0.8, 0] and [0, 1, 0, 0] -> exact count (2 + 1) / 2 = 1.5
    assert float(cs.strict_violation) == pytest.approx(0.0)
    assert cs.violation.requires_grad and not cs.strict_violation.requires_grad


def test_count_requires_per_pair_units():
    with pytest.raises(ValueError):
        SSAEConstrainedProblem(torch.device("cpu"), batch=2, hid=4, n_concepts=4,
                               sparsity_type="count", budget_units="fraction", target=1.0)
    with pytest.raises(ValueError):
        Cfg(emb=Path("x.h5"), sparsity_type="count")


def test_fraction_units_default_is_unchanged():
    h = torch.tensor([[0.5, 0.0, -0.2, 0.01]])
    cmp = SSAEConstrainedProblem(torch.device("cpu"), batch=1, hid=4, n_concepts=1,
                                 target=0.5, sparsity_type="step_l0", step_l0_threshold=0.05)
    cmp.level = 0.25
    torch.testing.assert_close(cmp._compute_sparsity_violation(h), torch.tensor(2 / 4 - 0.25))


# ------------------------------------------------------------------ unit input norm
@pytest.mark.parametrize("gate", ["signed_hard", "signed_batchtopk", "none"])
def test_unit_norm_codes_scale_and_sign(gate):
    model = _unit_model(gate)
    if gate == "signed_batchtopk":
        model.train()
        model(torch.randn(256, 12))
    model.eval()
    x = torch.randn(20, 12)
    with torch.no_grad():
        _, h = model(x)
        for c in (0.3, 2.0, 17.0):
            _, hc = model(c * x)
            torch.testing.assert_close(hc, c * h, rtol=1e-5, atol=1e-6)
            assert torch.equal(hc != 0, h != 0)
        _, hn = model(-x)
        torch.testing.assert_close(hn, -h)


def test_unit_norm_relative_loss_equals_unit_loss():
    model = _unit_model("signed_hard")
    x = torch.randn(30, 12) * torch.rand(30, 1) * 10
    x_hat, _ = model(x)
    rel = ((x_hat - x).pow(2).sum(1) / x.pow(2).sum(1)).mean()
    u = x / x.norm(dim=1, keepdim=True)
    u_hat, _ = model(u)
    unit = (u_hat - u).pow(2).sum(1).mean()
    torch.testing.assert_close(rel, unit, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(model.normed_input(x), x)  # reconstruction target is raw x


def test_unit_norm_rejects_code_norm_and_biases():
    with pytest.raises(ValueError):
        DictLinearAE(4, 2, "ln", input_norm="unit", zero_biases=True)
    with pytest.raises(ValueError):
        DictLinearAE(4, 2, "none", input_norm="unit", zero_biases=False)


def test_unit_norm_matches_posthoc_gate():
    rng = np.random.default_rng(0)
    E = rng.standard_normal((4, 10))
    D = rng.standard_normal((10, 4))
    model = DictLinearAE(10, 4, "none", activation="signed_hard", input_norm="unit",
                         zero_biases=True, signed_hard_tau=0.2)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.tensor(E))
        model.decoder.weight.copy_(torch.tensor(D))
    x = rng.standard_normal((25, 10)) * 3
    with torch.no_grad():
        _, h = model(torch.tensor(x, dtype=torch.float32))
    np.testing.assert_allclose(h.numpy(), posthoc_gate(E, x, 0.2), rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------ subspace solver
def _planted(n=800, d=16, k=3, seed=0, noise=1e-3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, k))
    A /= np.linalg.norm(A, axis=0)
    sup = rng.integers(0, 4, size=n)  # 0: {0}, 1: {1}, 2: {2}, 3: {0, 1}
    c = np.zeros((n, k))
    for i, s in enumerate(sup):
        idx = [0, 1] if s == 3 else [s]
        c[i, idx] = rng.standard_normal(len(idx))
    return c @ A.T + noise * rng.standard_normal((n, d)), c, A


def test_subspace_rows_unit_norm_convergence_and_deterministic_restarts():
    x, _, _ = _planted()
    E1, D1, obj1, objs1 = subspace_l0_batched(x, 3, restarts=3, steps=300, seed=4)
    E2, D2, obj2, objs2 = subspace_l0_batched(x, 3, restarts=3, steps=300, seed=4)
    np.testing.assert_allclose(E1, E2)
    assert objs1 == objs2 and obj1 == min(objs1)
    np.testing.assert_allclose(np.linalg.norm(D1, axis=0), 1.0, atol=1e-10)
    # D E is the projector onto the fitted subspace (exact reconstruction there)
    P = D1 @ E1
    np.testing.assert_allclose(P @ P, P, atol=1e-8)
    # The objective converges instead of drifting: with unit-norm rows it is bounded
    # below by r * log(eps_end) (Hadamard: |det W| <= 1) and settles once eps is fixed;
    # without the row normalisation the same objective drifts through that bound.
    trace = []
    subspace_l0_batched(x, 3, restarts=3, steps=1000, seed=4, trace=trace)
    tail = np.asarray(trace[-200:])
    bound = 3 * np.log(1e-3)
    assert np.all(np.isfinite(tail)) and np.all(tail >= bound)
    late, early = tail[-100:].mean(0), tail[:100].mean(0)
    assert np.all(np.abs(late - early) < 0.15 * np.abs(early))

    # Without unit-norm rows the objective is unbounded below when codes are exactly
    # sparse: scaling the true unmixing W by c lowers it by (r - E||code||_0) log c.
    x0, _, A = _planted(noise=0.0)
    xs = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    U = np.linalg.svd(xs.T @ xs)[0][:, :3]
    u = xs @ U
    W = np.linalg.inv(U.T @ A)
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    def obj(M):
        z = u @ M.T
        return -np.linalg.slogdet(M)[1] + 0.5 * np.log(z ** 2 + 1e-6).sum(1).mean()

    values = [obj(c * W) for c in (1.0, 10.0, 1e3, 1e6)]
    assert values[0] >= bound
    assert all(b < a for a, b in zip(values, values[1:])) and values[-1] < bound


def test_subspace_batched_matches_reference_solver():
    # In the smooth regime (eps >= 0.05) the vectorised restarts reproduce the
    # reference solver step for step; at eps -> 1e-3 Adam is chaotic, so any
    # reimplementation diverges in the last digits and only the atoms are compared.
    x, _, _ = _planted(seed=1)
    E_ref, D_ref, obj_ref = subspace_l0(x, 3, restarts=2, steps=80, seed=2, eps_end=0.05)
    E, D, obj, _ = subspace_l0_batched(x, 3, restarts=2, steps=80, seed=2, eps_end=0.05)
    assert obj == pytest.approx(obj_ref, rel=1e-9, abs=1e-10)
    np.testing.assert_allclose(E, E_ref, rtol=1e-7, atol=1e-9)
    _, D_ref, _ = subspace_l0(x, 3, restarts=2, steps=600, seed=2)
    _, D, _, _ = subspace_l0_batched(x, 3, restarts=2, steps=600, seed=2)
    cos = np.abs(D_ref.T @ D)
    assert np.sort(cos.max(1)).min() > 0.99


def test_subspace_recovers_planted_atoms_and_selection_is_label_free():
    x, _, A = _planted(n=2500)
    result = fit(x[:2000], x[2000:], r_max=8, restarts=4, steps=800, seed=0)
    assert result.r == 3
    cos = np.abs(A.T @ result.D)
    assert cos.max(axis=1).min() > 0.99
    assert result.r_curve is not None and len(result.r_curve.errors) == 8
    assert 0.0 <= result.tau <= 0.4


def test_export_loads_through_existing_loader_and_reproduces_codes(tmp_path):
    x, _, _ = _planted(n=600)
    result = fit(x[:500], x[500:], r=3, restarts=2, steps=200)
    run = export(result, tmp_path / "run")
    model = DictLinearAE.from_pretrained(run)
    assert model.input_norm_type == "unit" and model.zero_biases
    assert model.model_spec.canonical_name == "ssae_signed_hard_subspace"
    with torch.no_grad():
        _, h = model(torch.tensor(x[500:], dtype=torch.float32))
    np.testing.assert_allclose(h.numpy(), result.codes(x[500:]), rtol=1e-4, atol=1e-5)


def test_selection_rules_pick_expected_elbows():
    assert pick_beta_elbow([1, 2, 3, 4], [0.9, 0.2, 0.101, 0.1]) == 3
    rng = np.random.default_rng(0)
    A = np.linalg.qr(rng.standard_normal((20, 4)))[0]
    tr = rng.standard_normal((400, 4)) @ A.T
    ho = rng.standard_normal((200, 4)) @ A.T
    r, curve = select_r(tr, ho, r_max=10)
    assert r == 4 and curve.errors[3] < 1e-10
    tau, tcurve = select_tau(A.T, A, ho)
    assert tau in tcurve.xs
    err0, used0 = gated_error(A.T, A, ho, 0.0)
    assert err0 < 1e-10 and used0 == pytest.approx(4.0)


# ------------------------------------------------------------------ gated trainers
@pytest.mark.parametrize("gate", ["signed_hard", "signed_batchtopk"])
def test_gated_trainers_warm_start_and_respect_budget(gate):
    from ssae.gated_trainer import train

    x, _, A = _planted(n=1500)
    result = fit(x[:1200], x[1200:], r=3, restarts=2, steps=400)
    out = train(x[:1200], gate=gate, beta=1.3, tau=result.tau, init=(result.E, result.D),
                steps=200, lr=1e-3, heldout=x[1200:], device="cpu")
    assert out.codes_per_pair <= 1.3 + 0.35
    cos = np.abs(A.T @ out.D)
    assert cos.max(axis=1).min() > 0.95
    np.testing.assert_allclose(np.linalg.norm(out.D, axis=0), 1.0, atol=1e-5)


# ------------------------------------------------------------------ defaults unchanged
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "8debb1"),
        (dict(sparsity_type="l1", norm="ln", input_norm="rms", zero_biases=True,
              augment_reverse_diffs=True, max_updates=105000, schedule_unit="updates",
              renorm_updates=1, target=0.005), "4086d0"),
        (dict(sparsity_type="sigmoid_l0", sigmoid_l0_temperature=0.001, target=0.005), "f11e3d"),
        (dict(sparsity_type="step_l0", step_l0_surrogate="active_only", target=0.005), "1a4a2e"),
        (dict(activation="topk", topk_k=72), "d43461"),
    ],
)
def test_historical_run_hashes_are_unchanged(kwargs, expected):
    assert hash_cfg(Cfg(emb=Path("x.h5"), **kwargs)) == expected


def test_v4_flags_change_run_identity():
    base = dict(input_norm="unit", norm="none", zero_biases=True)
    a = hash_cfg(Cfg(emb=Path("x.h5"), activation="signed_hard", signed_hard_tau=0.05, **base))
    b = hash_cfg(Cfg(emb=Path("x.h5"), activation="signed_hard", signed_hard_tau=0.1, **base))
    c = hash_cfg(Cfg(emb=Path("x.h5"), activation="signed_hard", signed_hard_tau=0.05,
                     sparsity_type="count", budget_units="per_pair", target=2.0, **base))
    assert len({a, b, c}) == 3


def test_default_model_forward_is_unchanged():
    torch.manual_seed(0)
    model = DictLinearAE(6, 4, "ln", input_norm="none")
    x = torch.randn(5, 6)
    x_hat, h = model(x)
    expected_h = model.enc_norm(model.encoder(x - model.decoder.bias))
    torch.testing.assert_close(h, expected_h)
    torch.testing.assert_close(x_hat, model.decoder(expected_h))
    assert model.last_unit_preactivation is None


def test_explicit_inits_objective_helper_and_energy_tau_rule():
    from ssae.selection import pick_tau_energy, removed_subspace_energy
    from ssae.subspace_trainer import pca_ica_inits, subspace_objective

    x, _, A = _planted(n=1200)
    E, D, obj, objs, det = subspace_l0_batched(x, 3, restarts=2, steps=300, seed=1,
                                               init_Ws=pca_ica_inits(x, 3), return_details=True)
    assert len(objs) == 4 and obj == min(objs)
    assert subspace_objective(x, det["U"], det["W"]) == pytest.approx(obj, rel=1e-6, abs=1e-6)
    # an explicit init alone, with no steps, keeps its own objective
    W0 = det["W_all"][0]
    _, _, obj0, _ = subspace_l0_batched(x, 3, restarts=0, steps=0, init_Ws=[W0])
    assert obj0 == pytest.approx(subspace_objective(x, det["U"], W0), rel=1e-6, abs=1e-6)
    assert removed_subspace_energy(E, D, x, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert pick_tau_energy([0.0, 0.05, 0.1, 0.2], [0.0, 0.01, 0.019, 0.05]) == 0.1
    assert pick_tau_energy([0.0, 0.05, 0.1], [0.0, 0.03, 0.01]) == 0.0


def test_sign_merge_pairs_opposite_latents_only():
    from experiments.v4.methods import apply_merge, sign_merge

    rng = np.random.default_rng(0)
    d = rng.standard_normal((8, 3))
    atoms = np.concatenate([d, -d[:, :1], rng.standard_normal((8, 1))], axis=1)  # latent 3 = -latent 0
    codes = np.abs(rng.standard_normal((50, 5)))
    plus, minus = sign_merge(codes, atoms)
    assert (plus[0], minus[0]) == (0, 3) and (minus[1:] == -1).all() and sorted(plus) == [0, 1, 2, 4]
    merged = apply_merge(codes, plus, minus)
    np.testing.assert_allclose(merged[:, 0], codes[:, 0] - codes[:, 3])


def test_sae_trainer_early_stopping_and_antipodal_codes_are_odd():
    from experiments.v4.sae_baselines import train_one

    rng = np.random.default_rng(1)
    A = np.linalg.qr(rng.standard_normal((16, 4)))[0]
    X = (rng.standard_normal((600, 4)) * (rng.random((600, 4)) < 0.3)) @ A.T
    X = np.concatenate([X, -X]).astype(np.float32)
    t = train_one(X[:1000], X[1000:], "topk", 2, antipodal=True, device="cpu", max_updates=600)
    assert t.info["updates"] <= 600 and len(t.info["curve"]) >= 1
    c = t.codes(X[:20], device="cpu")
    np.testing.assert_allclose(t.codes(-X[:20], device="cpu"), -c, atol=1e-5)
    assert t.decoder().shape == (16, 16)


def test_completeness_uses_fixed_log_base():
    from experiments.v4.metrics import completeness_base

    R = np.zeros((10, 2))
    R[0, 0] = R[1, 1] = 1.0
    assert completeness_base(R, 64) == pytest.approx(1.0, abs=1e-6)
    R2 = np.ones((64, 1))
    assert completeness_base(R2, 64) == pytest.approx(0.0, abs=1e-6)
