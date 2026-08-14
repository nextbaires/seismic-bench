"""Adaptive-bandwidth smoothed seismicity — the same idea as `poisson`, sharper.

The fixed-bandwidth baseline in `poisson.py` smooths the whole region with one
kernel width. California is not one region for this purpose: the creeping
section of the San Andreas and the Salton Trough produce events at rates far
above the Mojave or the Sierra foothills, so a single bandwidth is simultaneously
too wide where the catalog is dense — smearing a genuinely localised rate over
cells that have never had an event — and too narrow where it is sparse, leaving
cells at the background floor because nothing happened to be recorded within a
few tens of kilometres (per @nextbaires, #1).

The fix is Helmstetter, Kagan & Jackson (2007),
https://doi.org/10.1785/gssrl.78.1.78: give every event its own bandwidth, taken
from the distance to its own n-th nearest neighbour in the catalog. The kernel
then narrows automatically where events are dense and widens where they are
sparse, without anything being fit to the data being forecast. That paper is the
method and the California result; Werner, Helmstetter, Jackson & Kagan (2011),
https://doi.org/10.1785/0120090340, is a later evaluation of the same family of
models, and smoothed-seismicity forecasts of this kind are the ones that have
survived prospective testing in the CSEP testing centres (Zechar, Schorlemmer et
al. 2013, https://doi.org/10.1785/0120120259).

This is a separate model rather than a change to `poisson`, so the fixed- and
adaptive-bandwidth versions stay side by side on the leaderboard (#1).

What is deliberately identical to `poisson`
-------------------------------------------
The total forecast rate — training events per day, projected over the window —
and the uniform background blend. Only the spatial distribution differs, which
is what makes the comparison a clean test of the bandwidth choice: the N-test
should be roughly unchanged and any movement should show up in the S-test.

What is not fit
---------------
`n_neighbours` and `min_bandwidth_km` are constructor constants, not estimates.
Picking either by comparing scores on these windows would be the leak this
benchmark exists to prevent, so n is fixed inside the 2nd-to-6th-neighbour range
the proposer cites from that paper (#1) and stays there.

What it actually does on the committed sample
---------------------------------------------
It raises the total log-likelihood — about -13,257 to -12,268 over the 36
windows, better in 34 of them — leaves the N-test pass rate untouched at 69%, as
a purely spatial change should, and **drops the S-test pass rate from 83% to
33%**. The proposer's stated falsification criterion (#1) was that the S-test
pass rate should rise, so on this catalog it is not met, and that is reported
rather than tuned away. The gammas fail low, meaning the forecast is more
spatially concentrated than the events turn out to be: with a ~20 km grid and an
M >= 3.0 catalog whose nearest-neighbour distances are mostly below the 2 km
floor, most events end up smoothed over less than one cell, so the model
approaches the raw historical count map. Whether that survives on the fuller,
lower-threshold catalog the method was designed for is the open question — see
the pull request for #1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import erf

from ..bench.grid import EARTH_RADIUS_KM, Grid
from ..bench.split import Window

KM_PER_DEG_LAT = 111.19


class AdaptivePoissonBaseline:
    """Time-independent smoothed seismicity with a per-event kernel width.

    Parameters
    ----------
    n_neighbours
        Which nearest neighbour sets an event's bandwidth. Helmstetter, Kagan &
        Jackson (2007), https://doi.org/10.1785/gssrl.78.1.78, work with the 2nd
        to 6th; 3 is a fixed choice inside that range, not one tuned on these
        windows.
    min_bandwidth_km
        Floor on the bandwidth. Without it a tight aftershock cluster, whose
        members sit within a few hundred metres of each other, collapses to a
        delta function on one cell (per @nextbaires, #1).
    min_magnitude
        Events below this are dropped from training, and from the neighbour
        distances that set the bandwidths. Should be at or above the catalog's
        completeness magnitude; below it, both the rate and the apparent
        clustering are properties of the seismic network rather than the Earth.
    background_fraction
        Fraction of the total rate spread uniformly over every cell, exactly as
        in `PoissonBaseline` — a sharper kernel makes this more load-bearing,
        not less, since more cells end up far from any kernel's mass.
    """

    def __init__(
        self,
        *,
        n_neighbours: int = 3,
        min_bandwidth_km: float = 2.0,
        min_magnitude: float = 3.0,
        background_fraction: float = 0.01,
    ) -> None:
        if n_neighbours < 1:
            raise ValueError("n_neighbours must be at least 1")
        if min_bandwidth_km <= 0:
            raise ValueError("min_bandwidth_km must be positive")
        if not 0.0 <= background_fraction < 1.0:
            raise ValueError("background_fraction must be in [0, 1)")
        self.n_neighbours = int(n_neighbours)
        self.min_bandwidth_km = float(min_bandwidth_km)
        self.min_magnitude = float(min_magnitude)
        self.background_fraction = float(background_fraction)
        self.name = "poisson-adaptive"

    # -- pieces of the model -------------------------------------------------

    def bandwidths_km(self, longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
        """Distance from each event to its `n_neighbours`-th nearest neighbour.

        Public because it is the whole content of the model: a reviewer checking
        that the kernel really is narrower in the creeping section than in the
        Mojave wants this array, not the forecast it produces.

        Neighbours are found on the sphere rather than in a flat projection —
        the region spans twelve degrees of latitude, over which a projection
        fixed at the mean latitude misstates east-west distances by several per
        cent, which is a bandwidth error in exactly the sparse regions where the
        bandwidth is largest.
        """
        n_events = len(longitude)
        k = min(self.n_neighbours, n_events - 1)
        if k < 1:
            # A single event has no neighbour to measure against. The floor is
            # the only defensible width left, and the uniform background carries
            # the rest of the region.
            return np.full(n_events, self.min_bandwidth_km)

        lon = np.deg2rad(longitude)
        lat = np.deg2rad(latitude)
        xyz = EARTH_RADIUS_KM * np.column_stack(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        # The first neighbour of a point is itself, hence k + 1.
        chord, _ = cKDTree(xyz).query(xyz, k=k + 1)
        chord = np.atleast_2d(chord)[:, -1]

        great_circle = 2.0 * EARTH_RADIUS_KM * np.arcsin(
            np.clip(chord / (2.0 * EARTH_RADIUS_KM), 0.0, 1.0)
        )
        return np.maximum(great_circle, self.min_bandwidth_km)

    def _smoothed_field(
        self, longitude: np.ndarray, latitude: np.ndarray, grid: Grid
    ) -> np.ndarray:
        """Sum of one isotropic Gaussian kernel per event, integrated per cell.

        Each kernel is the 2-D Gaussian normalised over the plane, so every event
        contributes the same total weight however wide its kernel is. The kernel
        is Gaussian — the same shape `PoissonBaseline` uses — so that the only
        difference between the two models is the bandwidth.

        The kernel is *integrated* over each cell rather than evaluated at the
        centre, unlike `etas.py`. That is not a refinement here, it is required:
        adaptive bandwidths in a dense cluster fall well below the ~20 km cell
        size, and a sub-cell kernel sampled at cell centres would give an event
        sitting near a centre orders of magnitude more weight than an identical
        event sitting near a corner, which is an artefact of the grid rather
        than anything about the seismicity.

        In the local flat projection the 2-D Gaussian is separable, so the mass
        in a cell is a product of two one-dimensional integrals and the whole
        field is one matrix product rather than a pairwise distance calculation.
        """
        sigma = self.bandwidths_km(longitude, latitude)

        # Cell edges, in the same lat-major layout the forecast is returned in.
        lon_edges = grid.lon_min + np.arange(grid.n_lon + 1) * grid.cell_deg
        lat_edges = grid.lat_min + np.arange(grid.n_lat + 1) * grid.cell_deg
        km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.deg2rad(np.clip(latitude, -89.9, 89.9)))

        field = np.zeros(grid.shape, dtype=float)
        # Chunked so memory stays bounded on long catalogs — the intermediate is
        # (chunk x edges), not (events x cells).
        chunk = 20_000
        for lo in range(0, len(sigma), chunk):
            hi = lo + chunk
            scale = sigma[lo:hi, None] * np.sqrt(2.0)
            lon_km = (lon_edges[None, :] - longitude[lo:hi, None]) * km_per_deg_lon[lo:hi, None]
            lat_km = (lat_edges[None, :] - latitude[lo:hi, None]) * KM_PER_DEG_LAT

            # Mass between consecutive edges: the Gaussian CDF differenced.
            lon_mass = 0.5 * np.diff(erf(lon_km / scale), axis=1)
            lat_mass = 0.5 * np.diff(erf(lat_km / scale), axis=1)
            field += lat_mass.T @ lon_mass

        return field.ravel()

    # -- interface -----------------------------------------------------------

    def forecast(self, training_events: pd.DataFrame, grid: Grid, window: Window) -> np.ndarray:
        events = training_events
        if not events.empty:
            events = events.loc[events["magnitude"] >= self.min_magnitude]

        n_cells = grid.n_cells
        uniform = np.full(n_cells, 1.0 / n_cells)

        if events.empty:
            # No history at all. Forecast a token uniform rate rather than zero:
            # claiming certainty of no events is a stronger statement than the
            # evidence supports.
            return uniform * window.duration_days / 365.25

        smoothed = self._smoothed_field(
            events["longitude"].to_numpy(dtype=float),
            events["latitude"].to_numpy(dtype=float),
            grid,
        )

        total = smoothed.sum()
        spatial = uniform if total <= 0 else smoothed / total

        # Blend toward uniform so no cell is exactly zero.
        spatial = (1.0 - self.background_fraction) * spatial + self.background_fraction * uniform

        # Observed rate per day over the training period, projected forward.
        training_days = max(window.training_days, 1e-9)
        rate_per_day = len(events) / training_days

        return spatial * rate_per_day * window.duration_days

    def describe(self) -> dict[str, object]:
        return {
            "model": "AdaptivePoissonBaseline",
            "bandwidth": "adaptive, n-th nearest neighbour",
            "n_neighbours": self.n_neighbours,
            "min_bandwidth_km": self.min_bandwidth_km,
            "min_magnitude": self.min_magnitude,
            "background_fraction": self.background_fraction,
            "bandwidth_fit": False,
            "bandwidth_source": (
                "Helmstetter, Kagan & Jackson (2007), doi:10.1785/gssrl.78.1.78; "
                "n fixed inside their 2-6 range, not tuned on these windows"
            ),
            "time_dependent": False,
        }
