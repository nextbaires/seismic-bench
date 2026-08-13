"""Fetch a catalog, validate it, write it.

    python -m seismic_bench.ingest.cli --start 2010-01-01 --end 2024-01-01

Defaults cover the California benchmark region. Quarantined rows are written
alongside the catalog and their count is printed — a run reporting a large
quarantine is telling you something about the source, not about this code.
"""

from __future__ import annotations

import argparse
import sys

from ..bench.grid import CALIFORNIA
from ..contracts import CATALOG_CONTRACT
from ..lake import quarantine_dir, write_catalog
from . import usgs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m seismic_bench.ingest.cli",
        description="Fetch an earthquake catalog into the canonical schema.",
    )
    parser.add_argument("--start", default="2010-01-01", help="inclusive, ISO date")
    parser.add_argument("--end", default="2024-01-01", help="exclusive, ISO date")
    parser.add_argument("--min-magnitude", type=float, default=2.5)
    parser.add_argument(
        "--region",
        default="california",
        choices=["california", "global"],
        help="california uses the default benchmark grid bounds",
    )
    parser.add_argument("--chunk-days", type=int, default=90)
    args = parser.parse_args(argv)

    bounds = {}
    if args.region == "california":
        bounds = {
            "min_longitude": CALIFORNIA.lon_min,
            "max_longitude": CALIFORNIA.lon_max,
            "min_latitude": CALIFORNIA.lat_min,
            "max_latitude": CALIFORNIA.lat_max,
        }

    print(f"Fetching USGS {args.start} .. {args.end}, M>={args.min_magnitude}", file=sys.stderr)
    raw = usgs.fetch(
        args.start,
        args.end,
        min_magnitude=args.min_magnitude,
        chunk_days=args.chunk_days,
        **bounds,
    )
    print(f"Fetched {len(raw)} events", file=sys.stderr)

    clean, quarantined = CATALOG_CONTRACT.apply(raw)
    path = write_catalog(clean)

    print(f"Wrote {len(clean)} events -> {path}", file=sys.stderr)
    if not quarantined.empty:
        qpath = CATALOG_CONTRACT.write_quarantine(quarantined)
        reasons = quarantined["_reasons"].value_counts().to_dict()
        print(
            f"Quarantined {len(quarantined)} row(s) -> {qpath}\n  {reasons}",
            file=sys.stderr,
        )
    else:
        print(f"Nothing quarantined (would have gone to {quarantine_dir()})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
