"""Aggregate individual runs into the published leaderboard.

    python -m seismic_bench.leaderboard

Reads every `leaderboard/runs/*.json` and writes `leaderboard/results.json`,
which is what the website reads and what CI regenerates on merge.

Runs are only comparable if they used the same protocol, so entries are grouped
by a protocol fingerprint and ranked within a group. Two models scored on
different regions, magnitude thresholds, or window lengths do not share a
ranking — quietly ranking them together is how a benchmark starts lying.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "leaderboard" / "runs"
RESULTS_PATH = REPO_ROOT / "leaderboard" / "results.json"


def protocol_key(run: dict) -> str:
    """A human-readable fingerprint of the scoring protocol."""
    protocol = run["protocol"]
    grid = run["grid"]
    return (
        f"M>={protocol['min_magnitude']} "
        f"{protocol['test_days']}d-windows "
        f"{grid['cell_deg']}deg-cells "
        f"lon[{grid['lon_min']},{grid['lon_max']}] "
        f"lat[{grid['lat_min']},{grid['lat_max']}]"
    )


def collect(runs_dir: Path = RUNS_DIR) -> list[dict]:
    if not runs_dir.exists():
        return []
    runs = []
    for path in sorted(runs_dir.glob("*.json")):
        runs.append(json.loads(path.read_text()))
    return runs


def build(runs: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for run in runs:
        entry = {
            "model": run["model"],
            "total_log_likelihood": run["totals"]["total_log_likelihood"],
            "mean_log_likelihood": run["totals"]["mean_log_likelihood"],
            "mean_brier": run["totals"]["mean_brier"],
            "n_test_pass_rate": run["totals"]["n_test_pass_rate"],
            "s_test_pass_rate": run["totals"]["s_test_pass_rate"],
            "n_windows": run["totals"]["n_windows"],
            "total_forecast": run["totals"]["total_forecast"],
            "total_observed": run["totals"]["total_observed"],
            "parameters_fit": run["description"].get("parameters_fit"),
            "time_dependent": run["description"].get("time_dependent"),
            "description": run["description"],
        }
        groups.setdefault(protocol_key(run), []).append(entry)

    # Higher log-likelihood is better.
    for entries in groups.values():
        entries.sort(key=lambda e: e["total_log_likelihood"], reverse=True)
        for rank, entry in enumerate(entries, start=1):
            entry["rank"] = rank

    # Deliberately no timestamp and no library versions. This file is committed
    # and CI fails if it drifts from what the code produces, so anything that
    # varies between two identical runs would break that check on every push and
    # train everyone to ignore it. Provenance that does vary — when a run
    # happened, on which Python — stays in the per-run records under
    # leaderboard/runs/, which git does not track.
    return {
        "ranked_by": "total_log_likelihood, higher is better",
        "note": (
            "Entries are only comparable within a protocol group. Passing the "
            "consistency tests matters independently of rank: a model can rank "
            "first on likelihood while failing the N-test, which means it gets "
            "the spatial pattern right and the event count wrong."
        ),
        "protocols": [
            {"protocol": key, "entries": entries} for key, entries in sorted(groups.items())
        ],
    }


def main() -> int:
    runs = collect()
    if not runs:
        print(f"No runs found in {RUNS_DIR}. Run the benchmark first.")
        return 1

    results = build(runs)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")

    for group in results["protocols"]:
        print(f"\n{group['protocol']}")
        print(f"  {'rank':<5} {'model':<12} {'total LL':>12} {'N-test':>8} {'S-test':>8}")
        for entry in group["entries"]:
            print(
                f"  {entry['rank']:<5} {entry['model']:<12} "
                f"{entry['total_log_likelihood']:>12.2f} "
                f"{entry['n_test_pass_rate']:>7.0%} {entry['s_test_pass_rate']:>8.0%}"
            )
    print(f"\n-> {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
