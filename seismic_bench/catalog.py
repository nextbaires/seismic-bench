"""The canonical catalog schema.

Every ingester, whatever agency it reads from, produces this and only this. The
point is that a forecast model written against `seismic-bench` never learns the
quirks of ComCat or EMSC — it sees one table, and swapping the source out is not
a code change for anyone downstream.

Columns
-------
event_id      str      Source-qualified unique id, e.g. "usgs:ci38457511".
time          datetime UTC, timezone-aware. Origin time, not first-arrival time.
latitude      float    Degrees, [-90, 90].
longitude     float    Degrees, normalized to [-180, 180).
depth_km      float    Kilometres below sea level. Negative depths are real
                       (events above sea level in mountainous regions) and are
                       kept, not clipped.
magnitude     float    Preferred magnitude as reported by the source.
mag_type      str      "mw", "ml", "mb", ... lowercased. Not homogenized —
                       see the note below.
source        str      Agency key: "usgs", "emsc", ...

On magnitude homogenization
---------------------------
We deliberately do not convert magnitude types to a common scale. Conversion
relations are regional, time-dependent, and contested, and baking one in would
silently make every downstream forecast a forecast about our choice of
conversion. Models that care should filter on `mag_type` or apply their own
relation, and say which they used.
"""

from __future__ import annotations

import pandas as pd

SCHEMA: dict[str, str] = {
    "event_id": "string",
    "time": "datetime64[ns, UTC]",
    "latitude": "float64",
    "longitude": "float64",
    "depth_km": "float64",
    "magnitude": "float64",
    "mag_type": "string",
    "source": "string",
}

COLUMNS = list(SCHEMA)


def empty() -> pd.DataFrame:
    """A zero-row frame with the right dtypes, for the no-results path."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in SCHEMA.items()})


def normalize_longitude(lon: pd.Series) -> pd.Series:
    """Wrap to [-180, 180).

    Agencies disagree: ComCat uses [-180, 180), some regional catalogs use
    [0, 360). Left alone, a Pacific-crossing region silently loses half its
    events to a filter comparison.
    """
    return ((lon + 180.0) % 360.0) - 180.0


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a source-shaped frame to the canonical schema.

    Missing columns are an error, not a fill — a source that cannot supply depth
    should say so explicitly with a NaN column rather than have one invented here.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"cannot conform, missing columns: {missing}")

    out = df.loc[:, COLUMNS].copy()
    # Pinned to nanosecond resolution. pandas 2 infers unit from the input, so
    # ISO strings parse as microseconds and epoch-millisecond integers as
    # milliseconds — concatenating two sources then yields a mixed-resolution
    # column that compares badly against a ns Timestamp. The schema declares ns;
    # this makes that true rather than aspirational.
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce").astype(
        "datetime64[ns, UTC]"
    )
    for col in ("latitude", "longitude", "depth_km", "magnitude"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["longitude"] = normalize_longitude(out["longitude"])
    for col in ("event_id", "mag_type", "source"):
        out[col] = out[col].astype("string")
    out["mag_type"] = out["mag_type"].str.lower()

    return out.sort_values("time", kind="stable").reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated event_ids, keeping the last.

    Catalogs are revised: an event ingested during an aftershock sequence and
    re-ingested a month later will have a better-constrained magnitude the
    second time. Last-wins means a re-run improves the catalog rather than
    duplicating it.
    """
    return (
        df.sort_values("time", kind="stable")
        .drop_duplicates(subset="event_id", keep="last")
        .reset_index(drop=True)
    )
