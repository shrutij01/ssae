"""Code shared by more than one numbered experiment."""

import os
import tempfile
from pathlib import Path

# Result builders run in CI and on headless cluster nodes.  Set these before a
# report module imports Matplotlib.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ssae-matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")
