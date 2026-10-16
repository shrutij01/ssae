"""Submit the whole test phase of this study (only after PREREGISTRATION.md is committed), on at most 4 GPUs.

    python -m experiments.v5_activation.testrun submit

CPU: scoring (2 jobs), interchange exports. GPU lanes: (1) erasure of the 11 rows, then steering of the TruthfulQA rows;
(2) steering of the application rows, refusal, judge; (3, 4) interchange (first half of the shards); after lanes 1 and 2
the second half of the interchange shards runs on their GPUs.
"""

from __future__ import annotations

import argparse
import subprocess

from experiments.v5_activation.common import MODEL_TAG, REPO, ROWS, act_root, preregistration_committed



def sb(args: list[str]) -> str:
    return subprocess.run(["sbatch", "--parsable"] + args, capture_output=True, text=True, check=True).stdout.strip()


def submit() -> None:
    if not preregistration_committed():
        raise SystemExit("PREREGISTRATION.md (with the configs hash) is not committed at HEAD")
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run = str(REPO / "experiments/v5_activation/jobs/run.sh")
    log = lambda name: f"--output={act_root() / 'logs' / (name + '-%j.out')}"
    gpu = ["--partition=long", "--gres=gpu:1", "--constraint=lovelace|40gb|80gb", "--cpus-per-task=8", "--mem=64G"]
    cpu = ["--partition=long-cpu", "--cpus-per-task=8", "--mem=64G", "--time=12:00:00"]
    it = "google/gemma-2-2b-it"
    it_rows = [d for d, m, _, _ in ROWS if m == it]
    base_rows = [d for d, m, _, _ in ROWS if m != it]
    ids = {}
    ids["score_it"] = sb(cpu + ["--job-name=act-score-test-it", log("act-score-test-it"), run, sha,
                                "experiments.v5_activation.scoring", "--model", it, "--split", "test", "--dataset"] + it_rows)
    ids["score_2b"] = sb(cpu + ["--job-name=act-score-test-2b", log("act-score-test-2b"), run, sha,
                                "experiments.v5_activation.scoring", "--model", "google/gemma-2-2b", "--split", "test",
                                "--dataset"] + base_rows)
    # interchange: plan on the login node, exports on CPU
    import sys
    subprocess.run([sys.executable, "-m", "experiments.v5_activation.iia", "plan", "--split", "test"], check=True, cwd=REPO)
    ids["iia_prep"] = sb(["--partition=main-cpu", "--cpus-per-task=8", "--mem=48G", "--time=8:00:00",
                          "--job-name=act-iia-prep-test", log("act-iia-prep-test"), run, sha,
                          "experiments.v5_activation.iia", "prep", "--split", "test"])
    er_rows = [(d, m) for d, m, _, _ in ROWS if not d.startswith(("sycophancy", "refusal"))]
    lane1 = []
    for model in (it, "google/gemma-2-2b"):
        rows = [d for d, m in er_rows if m == model]
        lane1.append(sb(gpu + ["--time=12:00:00", f"--job-name=act-era-test-{MODEL_TAG[model][-2:]}",
                               log(f"act-era-test-{MODEL_TAG[model]}")]
                        + ([f"--dependency=afterany:{lane1[-1]}"] if lane1 else [])
                        + [run, sha, "experiments.v5_activation.panel", "erasure", "--model", model, "--split", "test",
                           "--dataset"] + rows))
    lane1.append(sb(gpu + ["--time=12:00:00", "--job-name=act-steer-tqa", log("act-steer-tqa"),
                           f"--dependency=afterany:{lane1[-1]}", run, sha, "experiments.v5_activation.panel", "steer",
                           "--split", "test", "--dataset", "truthful-qa-judge-v2", "truthful-qa-judge",
                           "truthful-qa-judge-noconst"]))
    lane2 = [sb(gpu + ["--time=12:00:00", "--job-name=act-steer-apps", log("act-steer-apps"), run, sha,
                       "experiments.v5_activation.panel", "steer", "--split", "test", "--dataset", "sycophancy-caa",
                       "sycophancy-swap", "refusal-caa", "refusal-swap"])]
    lane2.append(sb(gpu + ["--time=6:00:00", "--job-name=act-refusal", log("act-refusal"),
                           f"--dependency=afterok:{lane2[-1]}", run, sha, "experiments.v5_activation.panel", "refusal",
                           "--split", "test"]))
    lane2.append(sb(gpu + ["--time=3:00:00", "--job-name=act-judge", log("act-judge"), f"--dependency=afterok:{lane2[-1]}",
                           run, sha, "experiments.v5_activation.panel", "judge", "--split", "test"]))
    ids.update({"lane1": lane1, "lane2": lane2})
    print(ids)
    print("next: after the exports finish (job", ids["iia_prep"], "), run: python -m experiments.v5_activation.iia submit "
          f"--split test --jobs 2 --sha {sha}  (and 2 more with --dependency afterany:{lane1[-1]}:{lane2[-1]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["submit"])
    ap.parse_args()
    submit()


if __name__ == "__main__":
    main()
