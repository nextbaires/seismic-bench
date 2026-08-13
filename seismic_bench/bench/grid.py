"""Spatial binning.

A forecast in this benchmark is a rate per cell per window: an array of expected
event counts aligned to a fixed grid. Fixing the grid up front is what makes two
models comparable at all — otherwise a model can look better by choosing a
coarser binning, which shows up as a real improvement in the likelihood without
being one.

The grid is regular in longitude and latitude. That means cells are not equal
area: a 0.1 degree cell at 60 degrees north covers about half the ground of one
at the equator. This is the CSEP convention and it is kept for comparability,
but it matters for interpretation — a rate *density* comparison across latitudes
needs `cell_areas_km2`, which is provided for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class Grid:
    """A regular lon/lat grid over a rectangular region.

    Bounds are half-open on the upper edge (`[min, max)`) so that adjacent
    regions tile without double-counting an event on a shared boundary.
    """

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    cell_deg: float

    def __post_init__(self) -> None:
        if self.cell_deg <= 0:
            raise ValueError("cell_deg must be positive")
        if self.lon_min >= self.lon_max or self.lat_min >= self.lat_max:
            raise ValueError("region bounds must be increasing")

    @property
    def n_lon(self) -> int:
        return int(np.ceil((self.lon_max - self.lon_min) / self.cell_deg))

    @property
    def n_lat(self) -> int:
        return int(np.ceil((self.lat_max - self.lat_min) / self.cell_deg))

    @property
    def n_cells(self) -> int:
        return self.n_lon * self.n_lat

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_lat, self.n_lon)

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Centre coordinates of every cell, in flattened (lat-major) order."""
        lon = self.lon_min + (np.arange(self.n_lon) + 0.5) * self.cell_deg
        lat = self.lat_min + (np.arange(self.n_lat) + 0.5) * self.cell_deg
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        return lon_grid.ravel(), lat_grid.ravel()

    def cell_areas_km2(self) -> np.ndarray:
        """Area of each cell, flattened lat-major.

        Exact for a spherical Earth: integrating cos(lat) over the cell's
        latitude span rather than evaluating at the centre, which would be
        noticeably wrong for coarse cells at high latitude.
        """
        lat_edges = self.lat_min + np.arange(self.n_lat + 1) * self.cell_deg
        lat_lo = np.deg2rad(lat_edges[:-1])
        lat_hi = np.deg2rad(lat_edges[1:])
        dlon = np.deg2rad(self.cell_deg)
        band = EARTH_RADIUS_KM**2 * dlon * (np.sin(lat_hi) - np.sin(lat_lo))
        return np.repeat(band, self.n_lon)

    def index_of(self, longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
        """Flat cell index per point; -1 for points outside the region."""
        lon = np.asarray(longitude, dtype=float)
        lat = np.asarray(latitude, dtype=float)

        ix = np.floor((lon - self.lon_min) / self.cell_deg).astype(int)
        iy = np.floor((lat - self.lat_min) / self.cell_deg).astype(int)

        inside = (ix >= 0) & (ix < self.n_lon) & (iy >= 0) & (iy < self.n_lat)
        flat = iy * self.n_lon + ix
        return np.where(inside, flat, -1)

    def bin_events(self, events: pd.DataFrame) -> np.ndarray:
        """Observed counts per cell. Events outside the region are dropped."""
        if events.empty:
            return np.zeros(self.n_cells, dtype=float)
        idx = self.index_of(events["longitude"].to_numpy(), events["latitude"].to_numpy())
        idx = idx[idx >= 0]
        return np.bincount(idx, minlength=self.n_cells).astype(float)

    def contains(self, events: pd.DataFrame) -> pd.Series:
        """Mask of events falling inside the region."""
        if events.empty:
            return pd.Series([], dtype=bool, index=events.index)
        idx = self.index_of(events["longitude"].to_numpy(), events["latitude"].to_numpy())
        return pd.Series(idx >= 0, index=events.index)


#: California, roughly the CSEP RELM testing region. A convenient default
#: because the USGS catalog is dense, complete to a low magnitude, and long.
CALIFORNIA = Grid(lon_min=-125.0, lon_max=-113.0, lat_min=31.0, lat_max=43.0, cell_deg=0.2)
