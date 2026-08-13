"""Run a model across every window and write the result.

The protocol is fixed here rather than configurable per model, which is the
point: two entries on the leaderboard differ only in the model, never in how
they were scored.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines.base import validate_forecast
from ..lake import read_catalog
from . import metrics
from .grid import CALIFORNIA, Grid
from .split import rolling_windows

#: Built-in models. A submission adds a line here, or passes
#: `--model package.module:ClassName` to run without touching the registry.
REGISTRY: dict[str, str] = {
    "poisson": "seismic_bench.baselines.poisson:PoissonBaseline",
    "etas": "seismic_bench.baselines.etas:ETASBaseline",
}


def load_model(spec: str):
    """Resolve a registry key or a `module:ClassName` path to an instance."""
    target = REGISTRY.get(spec, spec)
    if ":" not in target:
        raise ValueError(
            f"unknown model {spec!r}. Use one of {sorted(REGISTRY)} "
            "or a 'package.module:ClassName' path."
        )
    module_name, class_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


@dataclass
class RunResult:
    model: str
    description: dict
    grid: dict
    protocol: dict
    per_window: list[dict] = field(default_factory=list)
    totals: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "description": self.description,
            "grid": self.grid,
            "protocol": self.protocol,
            "totals": self.totals,
            "per_window": self.per_window,
            "environment": self.environment,
        }


def run(
    model,
    catalog: pd.DataFrame,
    *,
    grid: Grid = CALIFORNIA,
    min_magnitude: float = 3.0,
    test_days: int = 30,
    min_training_days: int = 365 * 3,
    max_windows: int | None = None,
    seed: int | None = 0,
    verbose: bool = True,
) -> RunResult:
    """Score `model` on every rolling window the catalog supports.

    The catalog is filtered to the region and magnitude range *once*, before
    windows are cut, so training and test see identical filtering. Filtering
    inside the loop is a classic way to make training and test subtly different
    populations.
    """
    catalog = catalog.loc[grid.contains(catalog)]
    catalog = catalog.loc[catalog["magnitude"] >= min_magnitude]
    catalog = catalog.sort_values("time").reset_index(drop=True)

    windows = rolling_windows(
        catalog, test_days=test_days, min_training_days=min_training_days
    )
    if max_windows is not None:
        windows = windows[:max_windows]

    if not windows:
        raise ValueError(
            f"catalog spans {len(catalog)} events and yields no windows — it is "
            f"shorter than min_training_days ({min_training_days}) plus one "
            f"test window ({test_days} days)."
        )

    per_window: list[dict] = []
    for i, window in enumerate(windows, start=1):
        training = window.training_events(catalog)
        truth = window.test_events(catalog)

        rates = model.forecast(training, grid, window)
        rates = validate_forecast(rates, grid, getattr(model, "name", type(model).__name__))

        observed = grid.bin_events(truth)
        scores = metrics.score(rates, observed, seed=seed)

        per_window.append(
            {
                "window": i,
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                "n_training_events": int(len(training)),
                **scores.as_dict(),
            }
        )
        if verbose:
            print(
                f"  [{i}/{len(windows)}] {window.test_start:%Y-%m-%d}  "
                f"LL={scores.log_likelihood:10.2f}  "
                f"N_fore={scores.n_forecast:7.2f}  N_obs={scores.n_observed:4d}  "
                f"S-gamma={scores.s_test_gamma:.3f}",
                file=sys.stderr,
            )

    frame = pd.DataFrame(per_window)
    # Windows do not overlap, so the joint log-likelihood is the sum. The mean is
    # reported alongside because a run over 40 windows and one over 12 are
    # otherwise not comparable at a glance.
    totals = {
        "n_windows": len(windows),
        "total_log_likelihood": float(frame["log_likelihood"].sum()),
        "mean_log_likelihood": float(frame["log_likelihood"].mean()),
        "mean_brier": float(frame["brier"].mean()),
        "total_forecast": float(frame["n_forecast"].sum()),
        "total_observed": int(frame["n_observed"].sum()),
        # Fraction of windows passing each consistency test at conventional
        # levels: N-test two-sided at 5%, S-test one-sided at 5%.
        "n_test_pass_rate": float(
            ((frame["n_test_delta1"] >= 0.025) & (frame["n_test_delta2"] >= 0.025)).mean()
        ),
        "s_test_pass_rate": float((frame["s_test_gamma"].dropna() >= 0.05).mean())
        if frame["s_test_gamma"].notna().any()
        else float("nan"),
        "total_floored_cells": int(frame["n_floored_cells"].sum()),
    }

    return RunResult(
        model=getattr(model, "name", type(model).__name__),
        description=model.describe(),
        grid={
            "lon_min": grid.lon_min,
            "lon_max": grid.lon_max,
            "lat_min": grid.lat_min,
            "lat_max": grid.lat_max,
            "cell_deg": grid.cell_deg,
            "n_cells": grid.n_cells,
        },
        protocol={
            "min_magnitude": min_magnitude,
            "test_days": test_days,
            "min_training_days": min_training_days,
            "training": "expanding",
            "seed": seed,
            "catalog_events": int(len(catalog)),
            "catalog_start": catalog["time"].min().isoformat(),
            "catalog_end": catalog["time"].max().isoformat(),
        },
        per_window=per_window,
        totals=totals,
        environment={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "run_at": datetime.now(UTC).isoformat(),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m seismic_bench.bench.run",
        description="Score a forecast model on the benchmark protocol.",
    )
    parser.add_argument(
        "--model",
        default="poisson",
        help=f"registry key {sorted(REGISTRY)} or 'module:ClassName'",
    )
    parser.add_argument("--min-magnitude", type=float, default=3.0)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--min-training-days", type=int, default=365 * 3)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="cap the number of windows (for a quick check, not for a leaderboard entry)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the result JSON here (default: leaderboard/runs/<model>.json)",
    )
    args = parser.parse_args(argv)

    model = load_model(args.model)
    catalog = read_catalog()

    print(f"Running {args.model} on {len(catalog)} events", file=sys.stderr)
    result = run(
        model,
        catalog,
        min_magnitude=args.min_magnitude,
        test_days=args.test_days,
        min_training_days=args.min_training_days,
        max_windows=args.max_windows,
        seed=args.seed,
    )

    out = args.out or Path("leaderboard/runs") / f"{result.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2) + "\n")

    t = result.totals
    print(
        f"\n{result.model}: {t['n_windows']} windows  "
        f"total LL={t['total_log_likelihood']:.2f}  "
        f"N-test pass={t['n_test_pass_rate']:.0%}  "
        f"S-test pass={t['s_test_pass_rate']:.0%}\n"
        f"-> {out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
