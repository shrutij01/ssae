"""The study's remaining work as a task graph for ``taskq`` (idempotent: existing task ids are kept).

    python -m experiments.v6_ravel.plan

Queues: ``llama`` and ``gemma`` (3 GPU workers each, the user's 6-GPU cap split between the phases), ``cpu``
(feature selection, layer choice). Priorities: lower runs first.
"""

from __future__ import annotations

from experiments.v6_ravel.common import ENTITY_TYPES
from experiments.v6_ravel.taskq import add

LF = ["pca", "pca_delta", "ica_act", "ica_delta", "ssae_full", "ssae_L0", "ssae_L1", "ssae_A"]
FIT_PRIO = {"ssae_A": 8, "ssae_L0": 9, "ssae_L1": 9, "ssae_full": 10, "ica_delta": 11, "ica_act": 11,
            "pca_delta": 11, "pca": 11}
SHORT = {"city": "city", "nobel_prize_winner": "nobel", "verb": "verb", "physical_object": "obj",
         "occupation": "occ"}
R = "experiments.v6_ravel.run"
S = "experiments.v6_ravel.sae"
B = "experiments.v6_ravel.baselines"
P = "experiments.v6_ravel.pipeline"
G = "experiments.v6_ravel.generate"


def label_free(q: str, model: str, e: str, inst_dep: list[str], tag: str) -> list[str]:
    """fit -> select (cpu) -> dev per method; returns the dev task ids."""
    devs = []
    for m in LF:
        fit = f"{tag}-fit-{SHORT[e]}-{m}"
        add(q, fit, FIT_PRIO[m], inst_dep, [R, "fit", "--loop", "--model", model, "--entities", e, "--methods", m])
        sel = f"{tag}-sel-{SHORT[e]}-{m}"
        add("cpu", sel, 20, [fit], [R, "select", "--loop", "--model", model, "--entities", e, "--methods", m, "--jobs", "6"])
        dev = f"{tag}-dev-{SHORT[e]}-{m}"
        add(q, dev, 30, [sel], [R, "dev", "--loop", "--model", model, "--entities", e, "--methods", m])
        devs.append(dev)
        if m == "pca":
            rel = f"{tag}-dev-{SHORT[e]}-pca_rel"
            add(q, rel, 31, [sel], [R, "dev", "--loop", "--model", model, "--entities", e, "--methods", "pca_rel"])
            devs.append(rel)
    return devs


def refs(q, model, e, inst_dep, tag):
    a, b = f"{tag}-refs-val-{SHORT[e]}", f"{tag}-refs-test-{SHORT[e]}"
    add(q, a, 25, inst_dep, [R, "refs", "--model", model, "--entities", e, "--part", "val"])
    add(q, b, 40, inst_dep, [R, "refs", "--model", model, "--entities", e, "--part", "test"])
    return [a, b]


def tests(q, model, e, deps, tag, ssae_coord=True):
    t = f"{tag}-test-{SHORT[e]}"
    add(q, t, 45, deps, [R, "test", "--model", model, "--entities", e, "--methods", ",".join(LF + ["pca_rel"])])
    out = [t]
    if ssae_coord:
        c = f"{tag}-testcoord-{SHORT[e]}"
        add(q, c, 60, deps, [R, "test", "--coord", "--model", model, "--entities", e, "--methods",
                             "ssae_full,ssae_A,ssae_L0,ssae_L1"])
        out.append(c)
    return out


def llama() -> None:
    q, tag, model = "llama", "L", "llama2_7b"
    add(q, "L-gen-city", 1, [], [G, "--model", model, "--entities", "city", "--reverse"])
    add(q, "L-gen-rest", 2, [], [G, "--model", model, "--entities", "verb", "--reverse"])
    add(q, "L-pipe-city", 3, ["L-gen-city"], [P, "--model", model, "--entities", "city", "--steps",
                                              "filter,wiki,build,extract"])
    add(q, "L-pipe-rest", 4, ["L-gen-rest"], [P, "--model", model, "--entities",
                                              "physical_object,nobel_prize_winner,verb", "--steps",
                                              "filter,wiki,build,extract"])
    inst = {"occupation": [], "city": ["L-pipe-city"], "physical_object": ["L-pipe-rest"],
            "nobel_prize_winner": ["L-pipe-rest"], "verb": ["L-pipe-rest"]}
    final = []
    for e in ["occupation", "city", "physical_object", "nobel_prize_winner", "verb"]:
        devs = label_free(q, model, e, inst[e], tag)
        rr = refs(q, model, e, inst[e], tag)
        final += tests(q, model, e, devs + rr, tag)
    # parity: the paper's SAE recipe on city (layer 16)
    add(q, "L-sae-acts-city", 6, [], [S, "acts", "--model", model, "--entities", "city"])
    add(q, "L-sae-train-city", 7, ["L-sae-acts-city"], [S, "train", "--model", model, "--entities", "city"])
    add("cpu", "L-sae-sel-city", 20, ["L-sae-train-city", "L-pipe-city"], [S, "select", "--model", model,
                                                                          "--entities", "city"])
    add(q, "L-sae-dev-city", 30, ["L-sae-sel-city"], [S, "dev", "--model", model, "--entities", "city"])
    add(q, "L-sae-test-city", 45, ["L-sae-dev-city"], [S, "test", "--model", model, "--entities", "city"])


def gemma() -> None:
    q, tag, model = "gemma", "G", "gemma2_2b"
    add(q, "G-gen-rest", 1, [], [G, "--model", model, "--entities", "verb", "--reverse", "--batch", "512"])
    add(q, "G-pipe-rest", 2, ["G-gen-rest"], [P, "--model", model, "--entities",
                                              "physical_object,nobel_prize_winner,verb", "--steps", "filter,wiki,build",
                                              "--train"])
    for e in ENTITY_TYPES:
        s = SHORT[e]
        inst = [] if e in ("city", "occupation") else ["G-pipe-rest"]
        add(q, f"G-sweep-{s}", 3, inst, [B, "sweep", "--model", model, "--entities", e, "--layers", "4,6,8,10,13,16"])
        add("cpu", f"G-choose-{s}", 1, [f"G-sweep-{s}"], [B, "choose", "--model", model, "--entities", e,
                                                        "--layers", "4,6,8,10,13,16"])
        ch = [f"G-choose-{s}"]
        add(q, f"G-extract-{s}", 4, ch, [P, "--model", model, "--entities", e, "--steps", "extract"])
        ex = [f"G-extract-{s}"]
        # counterfactual baselines
        trains = []
        for m, pr in (("das", 12), ("mdas", 13), ("dbm", 14)):
            add(q, f"G-bl-{m}-{s}", pr, ch, [B, "train", "--model", model, "--entities", e, "--methods", m])
            trains.append(f"G-bl-{m}-{s}")
        add(q, f"G-bldev-{s}", 30, trains, [B, "dev", "--model", model, "--entities", e, "--methods", "das,mdas,dbm"])
        add(q, f"G-bltest-{s}", 45, [f"G-bldev-{s}"], [B, "test", "--model", model, "--entities", e, "--methods",
                                                      "das,mdas,dbm"])
        # label-free
        devs = label_free(q, model, e, ex, tag)
        rr = refs(q, model, e, ex, tag)
        tests(q, model, e, devs + rr, tag)
        # SAE recipe and GemmaScope
        add(q, f"G-sae-acts-{s}", 6, ch, [S, "acts", "--model", model, "--entities", e])
        add(q, f"G-sae-train-{s}", 7, [f"G-sae-acts-{s}"], [S, "train", "--model", model, "--entities", e])
        add("cpu", f"G-sae-sel-{s}", 20, [f"G-sae-train-{s}"] + ex, [S, "select", "--model", model, "--entities", e])
        add(q, f"G-sae-dev-{s}", 30, [f"G-sae-sel-{s}"], [S, "dev", "--model", model, "--entities", e])
        add(q, f"G-sae-test-{s}", 45, [f"G-sae-dev-{s}"], [S, "test", "--model", model, "--entities", e])
        add("cpu", f"G-gs-sel-{s}", 21, ex, [S, "select", "--gemmascope", "--model", model, "--entities", e])
        add(q, f"G-gs-dev-{s}", 30, [f"G-gs-sel-{s}"], [S, "dev", "--gemmascope", "--model", model, "--entities", e])
        add(q, f"G-gs-test-{s}", 45, [f"G-gs-dev-{s}"], [S, "test", "--gemmascope", "--model", model, "--entities", e])


SSAE = ["ssae_full", "ssae_A", "ssae_L0", "ssae_L1"]


def fairness() -> None:
    """The user's fairness addition (2026-10-02): every dictionary method also as (a) the paper's SAE protocol
    (<m>__enc, the method's own selection) and (b) the rotation of its orthonormalised decoder directions (<m>__rot,
    selection on the rotated coordinates); SAE recipe / GemmaScope get (b) (--rot), their existing rows are (a)."""
    import time
    from experiments.v6_ravel.taskq import Q, done
    for q, tag, model, ents in (("llama", "L", "llama2_7b", ["occupation", "city", "physical_object",
                                                               "nobel_prize_winner", "verb"]),
                                ("gemma", "G", "gemma2_2b", ENTITY_TYPES)):
        for e in ents:
            s = SHORT[e]
            devs = []
            for m in SSAE:
                add(q, f"{tag}-dev-{s}-{m}__enc", 32, [f"{tag}-sel-{s}-{m}"],
                    [R, "dev", "--loop", "--model", model, "--entities", e, "--methods", f"{m}__enc"])
                add("cpu", f"{tag}-sel-{s}-{m}__rot", 22, [f"{tag}-fit-{s}-{m}"],
                    [R, "select", "--loop", "--model", model, "--entities", e, "--methods", f"{m}__rot", "--jobs", "6"])
                add(q, f"{tag}-dev-{s}-{m}__rot", 32, [f"{tag}-sel-{s}-{m}__rot"],
                    [R, "dev", "--loop", "--model", model, "--entities", e, "--methods", f"{m}__rot"])
                devs += [f"{tag}-dev-{s}-{m}__enc", f"{tag}-dev-{s}-{m}__rot"]
            refs_ = [f"{tag}-refs-val-{s}", f"{tag}-refs-test-{s}"]
            meths = ",".join(f"{m}__{v}" for m in SSAE for v in ("enc", "rot"))
            for sfx, pr in (("", 46), ("-2", 47)):
                add(q, f"{tag}-testdict-{s}{sfx}", pr, devs + refs_,
                    [R, "test", "--model", model, "--entities", e, "--methods", meths])
            # the error-preserving coordinate swap (records only) is replaced by (a)
            for sfx in ("",):
                tid = f"{tag}-testcoord-{s}{sfx}"
                if not done(tid):
                    (Q / "done").mkdir(parents=True, exist_ok=True)
                    (Q / "done" / tid).write_text('{"skipped": "replaced by the (a) rows, DEVIATIONS #11", "t": %d}'
                                                  % time.time())
    # SAE recipe / GemmaScope (b)
    add("cpu", "L-sae-rotsel-city", 21, ["L-sae-train-city", "L-pipe-city"],
        [S, "select", "--rot", "--model", "llama2_7b", "--entities", "city"])
    add("llama", "L-sae-rotdev-city", 31, ["L-sae-rotsel-city"], [S, "dev", "--rot", "--model", "llama2_7b",
                                                                   "--entities", "city"])
    add("llama", "L-sae-rottest-city", 46, ["L-sae-rotdev-city"], [S, "test", "--rot", "--model", "llama2_7b",
                                                                    "--entities", "city"])
    for e in ENTITY_TYPES:
        s = SHORT[e]
        add("cpu", f"G-sae-rotsel-{s}", 21, [f"G-sae-train-{s}", f"G-extract-{s}"],
            [S, "select", "--rot", "--model", "gemma2_2b", "--entities", e])
        add("gemma", f"G-sae-rotdev-{s}", 31, [f"G-sae-rotsel-{s}"], [S, "dev", "--rot", "--model", "gemma2_2b",
                                                                       "--entities", e])
        add("gemma", f"G-sae-rottest-{s}", 46, [f"G-sae-rotdev-{s}"], [S, "test", "--rot", "--model", "gemma2_2b",
                                                                        "--entities", e])
        add("cpu", f"G-gs-rotsel-{s}", 22, [f"G-extract-{s}"],
            [S, "select", "--rot", "--gemmascope", "--model", "gemma2_2b", "--entities", e])
        add("gemma", f"G-gs-rotdev-{s}", 31, [f"G-gs-rotsel-{s}"], [S, "dev", "--rot", "--gemmascope", "--model",
                                                                     "gemma2_2b", "--entities", e])
        add("gemma", f"G-gs-rottest-{s}", 46, [f"G-gs-rotdev-{s}"], [S, "test", "--rot", "--gemmascope", "--model",
                                                                      "gemma2_2b", "--entities", e])


D = "experiments.v6_ravel.dfeat"


def dfeat() -> None:
    """The decoder featurizer (PREREGISTRATION Addendum 3): fits on random pairs of the SAE recipe's Wikipedia
    activations (3 fit processes per GPU), selection (cpu), dev, test; a "-2" helper shares each dev / test task."""
    add("llama", "DL-wacts", 2, [], [S, "acts", "--model", "llama2_7b", "--entities",
                                    "nobel_prize_winner,verb,physical_object,occupation"])
    for q, tag, model in (("llama", "DL", "llama2_7b"), ("gemma", "DG", "gemma2_2b")):
        for e in ENTITY_TYPES:
            s = SHORT[e]
            dep = ["DL-wacts"] if (model == "llama2_7b" and e != "city") else []
            add(q, f"{tag}-fit-{s}", 5, dep, [D, "fit", "--model", model, "--entities", e, "--procs", "3"])
            add("cpu", f"{tag}-sel-{s}", 20, [f"{tag}-fit-{s}"], [D, "select", "--model", model, "--entities", e,
                                                                   "--jobs", "30"])
            for sfx, pr in (("", 30), ("-2", 31)):
                add(q, f"{tag}-dev-{s}{sfx}", pr, [f"{tag}-sel-{s}"], [D, "dev", "--model", model, "--entities", e])
            for sfx, pr in (("", 45), ("-2", 46)):
                add(q, f"{tag}-test-{s}{sfx}", pr, [f"{tag}-dev-{s}", f"{tag}-dev-{s}-2"],
                    [D, "test", "--model", model, "--entities", e])


SR = "experiments.v6_ravel.selrules"


def selection() -> None:
    """The selection-rule appendix (Addendum 3 item 7), after items 1-6: the file's implementation checks first (cpu),
    then per model and entity type the scores / agreement / k-curves (GPU, after that type's test runs), then plots."""
    add("cpu", "DS-checks", 15, ["DG-fit-occ", "DG-fitL1-occ"], [SR, "checks"])
    runs = []
    for q, tag, model in (("llama", "DL", "llama2_7b"), ("gemma", "DG", "gemma2_2b")):
        for e in ENTITY_TYPES:
            s = SHORT[e]
            tid = f"DS-{tag[1]}-{s}"
            add(q, tid, 60, ["DS-checks", f"{tag}-test-{s}", f"{tag}-test-{s}-2"],
                [SR, "run", "--model", model, "--entities", e])
            runs.append(tid)
    add("cpu", "DS-plots", 70, runs, [SR, "plots"])


if __name__ == "__main__":
    import sys
    if "--fairness" in sys.argv:
        fairness()
    elif "--dfeat" in sys.argv:
        dfeat()
    elif "--selection" in sys.argv:
        selection()
    else:
        llama()
        gemma()
