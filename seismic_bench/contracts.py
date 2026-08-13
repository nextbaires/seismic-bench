"""Row-level validation with quarantine, at the ingest boundary.

Adapted from the contract/dead-letter pattern in an earlier data-engineering
project of mine. The reasoning transfers directly to seismic catalogs, which are
messy in exactly the way this pattern is for: a handful of events in a
million-row catalog will have a null depth, a magnitude of -9.9 standing in for
"unknown", or coordinates at (0, 0) from a failed location solve.

Two obvious responses are both wrong. Failing the whole load on a few bad rows
means one junk event blocks a year of ingest. Silently letting them through means
a magnitude of -9.9 quietly enters somebody's b-value fit.

So the batch is split: clean rows continue, bad rows go to `_quarantine/` tagged
with which rule they broke, and the run reports the count. The failure stays
visible without being fatal, and the quarantine file is a browsable record of
what your source actually sends you.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .lake import quarantine_dir


@dataclass(frozen=True)
class Rule:
    """A named predicate. `ok` returns a row-wise mask of *valid* rows."""

    name: str
    ok: Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Contract:
    table: str
    rules: tuple[Rule, ...]

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split into (clean, quarantined).

        Quarantined rows carry a `_reasons` column listing every rule they broke,
        semicolon-separated — not just the first, because an event with both a
        null magnitude and a null depth is a different kind of broken from one
        with only a null magnitude, and you want to see that in the file.
        """
        if df.empty:
            return df.copy(), df.copy()

        reasons = pd.Series([""] * len(df), index=df.index, dtype="object")
        for rule in self.rules:
            broke = ~rule.ok(df).fillna(False).astype(bool)
            reasons.loc[broke] = (reasons.loc[broke] + ";" + rule.name).str.lstrip(";")

        is_bad = reasons != ""
        clean = df.loc[~is_bad].copy()
        quarantined = df.loc[is_bad].copy()
        if not quarantined.empty:
            quarantined["_reasons"] = reasons.loc[is_bad]

        return clean.reset_index(drop=True), quarantined.reset_index(drop=True)

    def write_quarantine(self, quarantined: pd.DataFrame, root: Path | None = None) -> Path | None:
        if quarantined.empty:
            return None
        out_dir = quarantine_dir(root)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.table}.parquet"
        quarantined.to_parquet(path, index=False)
        return path


def _finite(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").notna()


def _between(df: pd.DataFrame, col: str, lo: float, hi: float) -> pd.Series:
    v = pd.to_numeric(df[col], errors="coerce")
    return v.between(lo, hi, inclusive="both")


#: The catalog contract.
#:
#: Bounds are deliberately generous — this rejects impossible values, not
#: implausible ones. Screening out unusual-but-real events here would be a
#: scientific decision disguised as a data-quality one, and it belongs in a
#: model's own filtering where it can be argued about, not in the ingest layer
#: where it is invisible.
CATALOG_CONTRACT = Contract(
    table="catalog",
    rules=(
        Rule("event_id_missing", lambda d: d["event_id"].notna() & (d["event_id"] != "")),
        Rule("time_unparseable", lambda d: d["time"].notna()),
        Rule("latitude_out_of_range", lambda d: _between(d, "latitude", -90, 90)),
        Rule("longitude_out_of_range", lambda d: _between(d, "longitude", -180, 180)),
        Rule("magnitude_missing", lambda d: _finite(d, "magnitude")),
        # -2 admits microseismicity from dense local networks; 10.0 is above the
        # largest instrumentally recorded event (Mw 9.5, Chile 1960) with room to
        # spare. Anything outside is a sentinel value, not an earthquake.
        Rule("magnitude_implausible", lambda d: _between(d, "magnitude", -2.0, 10.0)),
        # 750 km is below the deepest recorded events (~700 km, Tonga/Fiji slabs).
        # Negative depths are legitimate and are NOT rejected.
        Rule(
            "depth_implausible",
            lambda d: _between(d, "depth_km", -10.0, 750.0) | d["depth_km"].isna(),
        ),
        # A location solve that fails often emits exactly (0, 0), in the Gulf of
        # Guinea. Real events there are rare enough that the tradeoff favours
        # quarantining; the file makes the decision auditable.
        Rule(
            "null_island",
            lambda d: ~((d["latitude"].abs() < 1e-6) & (d["longitude"].abs() < 1e-6)),
        ),
    ),
)
