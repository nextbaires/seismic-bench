"""USGS ComCat (earthquake.usgs.gov) ingestion.

The FDSN event web service caps a single response at 20,000 events, and returns
an error rather than truncating silently — which is the good behaviour, but it
means any query worth making has to be chunked. Chunking is by time, and the
window halves on overflow until each piece fits.

API docs: https://earthquake.usgs.gov/fdsnws/event/1/
"""

from __future__ import annotations

import time
from datetime import timedelta

import pandas as pd
import requests

from .. import catalog as cat

ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"

#: The service's hard cap. Requests above it return HTTP 400.
MAX_EVENTS_PER_REQUEST = 20_000

#: Polite pacing between requests. The service has no published rate limit;
#: this keeps a full-catalog backfill from looking like an attack.
REQUEST_DELAY_SECONDS = 0.34


def _request(params: dict, timeout: int, session: requests.Session) -> list[dict]:
    response = session.get(ENDPOINT, params=params, timeout=timeout)
    if response.status_code == 204:  # documented "no events matched"
        return []
    response.raise_for_status()
    return response.json().get("features", [])


def _parse(features: list[dict]) -> pd.DataFrame:
    """GeoJSON features to the canonical schema.

    Coordinates arrive as [longitude, latitude, depth_km] — longitude first,
    which is the opposite of how everyone says it out loud and a reliable source
    of transposed-coordinate bugs.
    """
    if not features:
        return cat.empty()

    rows = []
    for feature in features:
        properties = feature.get("properties") or {}
        coordinates = (feature.get("geometry") or {}).get("coordinates") or [None, None, None]
        rows.append(
            {
                "event_id": f"usgs:{feature.get('id')}",
                # `time` is epoch milliseconds, UTC.
                "time": pd.to_datetime(properties.get("time"), unit="ms", utc=True),
                "longitude": coordinates[0],
                "latitude": coordinates[1],
                "depth_km": coordinates[2],
                "magnitude": properties.get("mag"),
                "mag_type": properties.get("magType"),
                "source": "usgs",
            }
        )
    return cat.conform(pd.DataFrame(rows))


def fetch(
    start: str,
    end: str,
    *,
    min_magnitude: float = 2.5,
    min_latitude: float | None = None,
    max_latitude: float | None = None,
    min_longitude: float | None = None,
    max_longitude: float | None = None,
    chunk_days: int = 90,
    timeout: int = 60,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch events in [start, end), chunked to stay under the response cap.

    Chunks are half-open and consecutive, so an event on a chunk boundary is
    fetched exactly once. `deduplicate` runs at the end anyway, since ComCat
    occasionally returns the same event under a revised id.
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if start_ts >= end_ts:
        raise ValueError("start must be before end")

    bounds = {
        "minlatitude": min_latitude,
        "maxlatitude": max_latitude,
        "minlongitude": min_longitude,
        "maxlongitude": max_longitude,
    }
    bounds = {k: v for k, v in bounds.items() if v is not None}

    frames: list[pd.DataFrame] = []
    session = requests.Session()
    session.headers["User-Agent"] = "seismic-bench (+https://github.com/nextbaires/seismic-bench)"

    cursor = start_ts
    span = timedelta(days=chunk_days)

    while cursor < end_ts:
        chunk_end = min(cursor + span, end_ts)
        params = {
            "format": "geojson",
            "starttime": cursor.isoformat(),
            "endtime": chunk_end.isoformat(),
            "minmagnitude": min_magnitude,
            "orderby": "time-asc",
            **bounds,
        }

        try:
            features = _request(params, timeout, session)
        except requests.HTTPError as exc:
            # 400 here almost always means "too many events". Halve and retry
            # rather than making the caller guess a chunk size that works for
            # both a quiet decade and an aftershock sequence.
            too_large = exc.response is not None and exc.response.status_code == 400
            if too_large and span > timedelta(days=1):
                span = max(span // 2, timedelta(days=1))
                if verbose:
                    print(f"  response too large, narrowing chunk to {span.days}d")
                continue
            raise

        if len(features) >= MAX_EVENTS_PER_REQUEST and span > timedelta(days=1):
            span = max(span // 2, timedelta(days=1))
            if verbose:
                print(f"  hit the {MAX_EVENTS_PER_REQUEST} cap, narrowing chunk to {span.days}d")
            continue

        frames.append(_parse(features))
        if verbose:
            print(f"  {cursor:%Y-%m-%d} .. {chunk_end:%Y-%m-%d}: {len(features)} events")

        cursor = chunk_end
        time.sleep(REQUEST_DELAY_SECONDS)

    if not frames:
        return cat.empty()

    return cat.deduplicate(pd.concat(frames, ignore_index=True))
