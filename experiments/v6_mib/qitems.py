"""Queue-item helpers: run the current code revision from any queue worker (jobs/bootstrap.py, DEVIATIONS #21).

``snapshot()`` makes (once) the immutable snapshot of the committed HEAD revision that run.sh would make, and returns its
path; ``boot_module`` / ``boot_mib`` wrap a repository module or an MIB-side mib_run.py command as a module item that
any worker (also one started on an older revision) runs from that snapshot.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
REPO = Path(__file__).resolve().parents[2]


def snapshot() -> Path:
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    snap = ROOT / "code" / sha
    if not (snap / ".complete").exists():
        tmp = snap.with_name(snap.name + f".tmp.{os.getpid()}")
        tmp.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"git -C {REPO} archive {sha} -- . ':(exclude)results' | tar -x -C {tmp}", shell=True,
                       check=True)
        (tmp / ".complete").touch()
        os.replace(tmp, snap)
    return snap


def boot_module(snap: Path, module: str, args: list) -> dict:
    return {"cmd": "module", "module": "cProfile",
            "args": [str(snap / "experiments" / "v6_mib" / "jobs" / "bootstrap.py"), "module", module] + [str(x) for x in args]}


def boot_mib(snap: Path, cmd: str, **kw) -> dict:
    args = [str(snap / "experiments" / "v6_mib" / "jobs" / "bootstrap.py"), "mib", "mib_run.py", cmd]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return {"cmd": "module", "module": "cProfile", "args": args}
