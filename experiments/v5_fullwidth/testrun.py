"""Submit the whole test phase of this study (only after PREREGISTRATION.md is committed), on at most 4 GPUs.

    python -m experiments.v5_fullwidth.testrun submit

CPU: scoring (2 jobs), interchange exports. GPU lanes: (1) erasure of the 11 rows, then steering of the TruthfulQA rows;
(2) steering of the application rows, refusal, judge; (3, 4) interchange (first half of the shards); after lanes 1 and 2
the second half of the interchange shards runs on their GPUs.
"""

from __future__ import annotations

import argparse
import subprocess

from experiments.v5_fullwidth.common import MODEL_TAG, REPO, ROWS, fw_root, preregistration_committed



def sb(args: list[str]) -> str:
    return subprocess.run(["sbatch", "--parsable"] + args, capture_output=True, text=True, check=True).stdout.strip()


def submit() -> None:
    if not preregistration_committed():
        raise SystemExit("PREREGISTRATION.md (with the configs hash) is not committed at HEAD")
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run = str(REPO / "experiments/v5_fullwidth/jobs/run.sh")
    log = lambda name: f"--output={fw_root() / 'logs' / (name + '-%j.out')}"
    gpu = ["--partition=long", "--gres=gpu:1", "--exclude=cn-k001,cn-k003,cn-d001", "--constraint=lovelace|40gb|80gb", "--cpus-per-task=8", "--mem=64G"]
    cpu = ["--exclude=cn-k001,cn-k003,cn-d001", "--partition=long-cpu", "--cpus-per-task=8", "--mem=64G", "--time=12:00:00"]
    it = "google/gemma-2-2b-it"
    it_rows = [d for d, m, _, _ in ROWS if m == it]
    base_rows = [d for d, m, _, _ in ROWS if m != it]
    ids = {}
    ids["score_it"] = sb(cpu + ["--job-name=fw-score-test-it", log("fw-score-test-it"), run, sha,
                                "experiments.v5_fullwidth.scoring", "--model", it, "--split", "test", "--dataset"] + it_rows)
    ids["score_2b"] = sb(cpu + ["--job-name=fw-score-test-2b", log("fw-score-test-2b"), run, sha,
                                "experiments.v5_fullwidth.scoring", "--model", "google/gemma-2-2b", "--split", "test",
                                "--dataset"] + base_rows)
    # interchange: plan on the login node, exports on CPU
    import sys
    subprocess.run([sys.executable, "-m", "experiments.v5_fullwidth.iia", "plan", "--split", "test"], check=True, cwd=REPO)
    ids["iia_prep"] = sb(["--exclude=cn-k001,cn-k003,cn-d001", "--partition=main-cpu", "--cpus-per-task=8", "--mem=48G", "--time=8:00:00",
                          "--job-name=fw-iia-prep-test", log("fw-iia-prep-test"), run, sha,
                          "experiments.v5_fullwidth.iia", "prep", "--split", "test"])
    er_rows = [(d, m) for d, m, _, _ in ROWS if not d.startswith(("sycophancy", "refusal"))]
    lane1 = []
    for model in (it, "google/gemma-2-2b"):
        rows = [d for d, m in er_rows if m == model]
        lane1.append(sb(gpu + ["--time=12:00:00", f"--job-name=fw-era-test-{MODEL_TAG[model][-2:]}",
                               log(f"fw-era-test-{MODEL_TAG[model]}")]
                        + ([f"--dependency=afterany:{lane1[-1]}"] if lane1 else [])
                        + [run, sha, "experiments.v5_fullwidth.panel", "erasure", "--model", model, "--split", "test",
                           "--dataset"] + rows))
    lane1.append(sb(gpu + ["--time=12:00:00", "--job-name=fw-steer-tqa", log("fw-steer-tqa"),
                           f"--dependency=afterany:{lane1[-1]}", run, sha, "experiments.v5_fullwidth.panel", "steer",
                           "--split", "test", "--dataset", "truthful-qa-judge-v2", "truthful-qa-judge",
                           "truthful-qa-judge-noconst"]))
    lane2 = [sb(gpu + ["--time=12:00:00", "--job-name=fw-steer-apps", log("fw-steer-apps"), run, sha,
                       "experiments.v5_fullwidth.panel", "steer", "--split", "test", "--dataset", "sycophancy-caa",
                       "sycophancy-swap", "refusal-caa", "refusal-swap"])]
    lane2.append(sb(gpu + ["--time=6:00:00", "--job-name=fw-refusal", log("fw-refusal"),
                           f"--dependency=afterok:{lane2[-1]}", run, sha, "experiments.v5_fullwidth.panel", "refusal",
                           "--split", "test"]))
    lane2.append(sb(gpu + ["--time=3:00:00", "--job-name=fw-judge", log("fw-judge"), f"--dependency=afterok:{lane2[-1]}",
                           run, sha, "experiments.v5_fullwidth.panel", "judge", "--split", "test"]))
    ids.update({"lane1": lane1, "lane2": lane2})
    print(ids)
    print("next: after the exports finish (job", ids["iia_prep"], "), run: python -m experiments.v5_fullwidth.iia submit "
          f"--split test --jobs 2 --sha {sha}  (and 2 more with --dependency afterany:{lane1[-1]}:{lane2[-1]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["submit"])
    ap.parse_args()
    submit()


if __name__ == "__main__":
    main()
