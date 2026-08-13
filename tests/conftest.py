"""Shared fixtures.

Everything here works offline from the committed sample, so the whole suite runs
on a fresh clone with no network and no credentials. That is a deliberate
constraint: a benchmark whose tests need the internet is a benchmark whose CI
fails for reasons unrelated to the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seismic_bench.bench.grid import Grid
from seismic_bench.lake import read_catalog


@pytest.fixture(scope="session")
def sample_catalog() -> pd.DataFrame:
    """Real USGS events, California, 2014-2020, M>=3.0.

    Includes the July 2019 Ridgecrest sequence, which is the useful part: it is
    a genuine M7.1 mainshock with thousands of aftershocks, so anything that
    breaks on clustered seismicity breaks here.
    """
    return read_catalog()


@pytest.fixture(scope="session")
def small_grid() -> Grid:
    """A coarse grid, for tests where 3,600 cells would only be slow."""
    return Grid(lon_min=-122.0, lon_max=-116.0, lat_min=33.0, lat_max=38.0, cell_deg=1.0)


@pytest.fixture
def synthetic_events() -> pd.DataFrame:
    """A small catalog with known, hand-checkable positions."""
    return pd.DataFrame(
        {
            "event_id": [f"s{i}" for i in range(5)],
            "time": pd.to_datetime(
                [
                    "2020-01-01T00:00:00Z",
                    "2020-01-02T00:00:00Z",
                    "2020-01-03T00:00:00Z",
                    "2020-06-01T00:00:00Z",
                    "2021-01-01T00:00:00Z",
                ],
                utc=True,
            ),
            "latitude": [34.5, 34.5, 35.5, 36.5, 34.5],
            "longitude": [-118.5, -118.5, -119.5, -120.5, -118.5],
            "depth_km": [5.0, 10.0, 3.0, 8.0, 12.0],
            "magnitude": [4.0, 3.2, 5.1, 3.5, 6.0],
            "mag_type": ["mw"] * 5,
            "source": ["test"] * 5,
        }
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
