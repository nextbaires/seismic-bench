"""Where things live on disk.

A flat medallion layout: raw source pulls land in `bronze/`, the conformed and
validated catalog in `silver/`, and anything a benchmark run consumes in
`gold/`. Only the small committed sample under `data/sample/` is in version
control; everything else is generated and gitignored, so a clone is small and a
`git status` after a full ingest is still clean.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def data_root(root: Path | None = None) -> Path:
    """Resolve the data directory.

    Order: explicit argument, then `SEISMIC_BENCH_DATA`, then `<repo>/data`. The
    environment variable exists so CI and a laptop can point at different disks
    without a flag on every command.
    """
    if root is not None:
        return Path(root)
    env = os.environ.get("SEISMIC_BENCH_DATA")
    return Path(env) if env else REPO_ROOT / "data"


def bronze_dir(root: Path | None = None) -> Path:
    return data_root(root) / "bronze"


def silver_dir(root: Path | None = None) -> Path:
    return data_root(root) / "silver"


def gold_dir(root: Path | None = None) -> Path:
    return data_root(root) / "gold"


def quarantine_dir(root: Path | None = None) -> Path:
    return data_root(root) / "_quarantine"


def sample_catalog_path() -> Path:
    """The committed sample. Small enough for git, real enough to test on."""
    return REPO_ROOT / "data" / "sample" / "catalog.sample.parquet"


def catalog_path(root: Path | None = None) -> Path:
    return silver_dir(root) / "catalog.parquet"


def write_catalog(df: pd.DataFrame, root: Path | None = None) -> Path:
    path = catalog_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_catalog(root: Path | None = None) -> pd.DataFrame:
    """Read the full catalog, falling back to the committed sample.

    The fallback is what lets CI, and a newcomer who has just cloned, run the
    whole benchmark end to end without network access or an ingest step. It is
    also why every test in this repository can be run offline.
    """
    path = catalog_path(root)
    if path.exists():
        return pd.read_parquet(path)
    sample = sample_catalog_path()
    if sample.exists():
        return pd.read_parquet(sample)
    raise FileNotFoundError(
        f"No catalog at {path} and no sample at {sample}. "
        "Run `python -m seismic_bench.ingest.cli --help` to fetch one."
    )
