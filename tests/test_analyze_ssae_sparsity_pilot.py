import csv
import subprocess
from pathlib import Path

from scripts.analyze_ssae_sparsity_pilot import _METRICS, aggregate_seeds

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER = REPO_ROOT / "scripts" / "analyze_ssae_sparsity_pilot.py"


def _progress(epoch, *, recon, p, mean_abs, step=False):
    suffix = ""
    if step:
        suffix = "  rect_support 0.2  rect_active_coverage 0.25"
    return (
        f"ep {epoch:04d}  recon {recon}  violation 0  target 0.005  "
        f"constraint {p}  l0 1  p {p}  mean_abs {mean_abs}  dual 0.1"
        f"{suffix}\n"
    )


def test_analyzer_matches_by_achieved_p_not_configured_target(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    configs = [
        ("l1_low", "l1", "0.001", "NA", 0.01, 0.10),
        # This L1 run has the same numeric target as step-L0, but achieved
        # activity is deliberately far away.
        ("l1_same_target", "l1", "0.005", "NA", 0.20, 0.01),
        ("rect", "step_l0", "0.005", "0.1", 0.011, 0.20),
    ]
    manifest = tmp_path / "manifest.tsv"
    fields = [
        "index",
        "job_name",
        "norm",
        "estimator",
        "target",
        "bandwidth",
        "threshold",
        "seed",
        "job_id",
        "stdout",
        "stderr",
        "script",
    ]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, (name, estimator, target, bandwidth, p, recon) in enumerate(
            configs
        ):
            stdout = logs / f"{name}.out"
            stdout.write_text(
                _progress(
                    1998,
                    recon=recon,
                    p=p,
                    mean_abs=0.03,
                    step=estimator == "step_l0",
                )
                + _progress(
                    1999,
                    recon=recon,
                    p=p,
                    mean_abs=0.03,
                    step=estimator == "step_l0",
                )
            )
            writer.writerow(
                {
                    "index": index,
                    "job_name": name,
                    "norm": "none",
                    "estimator": estimator,
                    "target": target,
                    "bandwidth": bandwidth,
                    "threshold": "0.01",
                    "seed": "0",
                    "job_id": str(index),
                    "stdout": stdout,
                    "stderr": logs / f"{name}.err",
                    "script": tmp_path / f"{name}.sh",
                }
            )

    result = subprocess.run(
        [str(ANALYZER), str(manifest), "--tail", "2"],
        check=True,
        capture_output=True,
        text=True,
    )

    with (tmp_path / "summary.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rect = next(row for row in rows if row["job_name"] == "rect")

    assert rect["nearest_l1_job"] == "l1_low"
    assert float(rect["sparsity_target_mean"]) == 0.005
    assert float(rect["nearest_l1_p"]) == 0.01
    assert float(rect["delta_recon_vs_l1"]) == 0.1
    with (tmp_path / "aggregate.tsv").open(newline="") as handle:
        aggregates = list(csv.DictReader(handle, delimiter="\t"))
    rect_aggregate = next(
        row for row in aggregates if row["surrogate"] == "rectangle"
    )
    assert rect_aggregate["n_seeds"] == "1"
    assert float(rect_aggregate["sparsity_target_seed_mean"]) == 0.005
    assert float(rect_aggregate["p_seed_mean"]) == 0.011
    assert float(rect_aggregate["nearest_l1_p"]) == 0.01
    assert "complete: 3/3" in result.stdout


def test_seed_aggregation_keeps_schedule_and_dual_settings_separate():
    rows = []
    for schedule, dual_lr_div in (("100", "0.5"), ("500", "1.0")):
        row = {
            "status": "complete",
            "norm": "ln",
            "estimator": "step_l0",
            "surrogate": "rectangle",
            "target": "0.005",
            "bandwidth": "1.0",
            "schedule": schedule,
            "dual_lr_div": dual_lr_div,
            "seed": "0",
        }
        for metric in _METRICS:
            row[f"{metric}_mean"] = 0.1
        rows.append(row)

    aggregates = aggregate_seeds(rows)

    assert len(aggregates) == 2
    assert {
        (row["schedule"], row["dual_lr_div"]) for row in aggregates
    } == {("100", "0.5"), ("500", "1.0")}
