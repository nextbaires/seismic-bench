"""Smoothed time-independent seismicity — the baseline you have to beat.

The model is deliberately unsophisticated: count what happened in each cell
historically, smooth it spatially, and assume the future looks like the past at
the same rate. It knows nothing about aftershocks, nothing about stress
transfer, nothing about time.

It is here because it is genuinely hard to beat, and because a great deal of
published skill turns out to be this model in a costume. Smoothed seismicity is
a strong forecaster for the simple reason that earthquakes happen where
earthquakes have happened. If a new model does not beat this one out of sample,
it has not demonstrated anything, whatever else it can do.

The spatial smoothing is a fixed-bandwidth Gaussian, which is the crude version.
Adaptive bandwidth — wider kernels where data is sparse, following Helmstetter,
Kagan and Jackson (2007), https://doi.org/10.1785/gssrl.78.1.78 — is known to do
better and is not implemented here. That is a real, well-specified improvement
somebody should propose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from ..bench.grid import Grid
from ..bench.split import Window


class PoissonBaseline:
    """Time-independent smoothed-seismicity forecast.

    Parameters
    ----------
    smoothing_km
        Gaussian kernel bandwidth, in kilometres, converted to cells at the
        region's mean latitude. 50 km is a common default and is not tuned —
        tuning it on the test windows would be the exact leak this benchmark
        exists to prevent.
    min_magnitude
        Events below this are dropped from training. Should be at or above the
        catalog's completeness magnitude; below it, the apparent rate is a
        property of the seismic network rather than of the Earth.
    background_fraction
        Fraction of the total rate spread uniformly over every cell. Without it
        the model forecasts exactly zero everywhere it has never seen an event,
        which is both overconfident and, under the scorer's rate floor, a
        guaranteed way to look artificially bad the first time a quiet cell
        activates.
    """

    def __init__(
        self,
        *,
        smoothing_km: float = 50.0,
        min_magnitude: float = 3.0,
        background_fraction: float = 0.01,
    ) -> None:
        if not 0.0 <= background_fraction < 1.0:
            raise ValueError("background_fraction must be in [0, 1)")
        self.smoothing_km = float(smoothing_km)
        self.min_magnitude = float(min_magnitude)
        self.background_fraction = float(background_fraction)
        self.name = "poisson"

    def _sigma_cells(self, grid: Grid) -> tuple[float, float]:
        """Kernel bandwidth in cells, separately for latitude and longitude.

        A degree of longitude shrinks with latitude, so a kernel that is
        isotropic in kilometres is anisotropic in cells. Ignoring this stretches
        the kernel east-west at high latitude.
        """
        km_per_deg_lat = 111.19
        mean_lat = np.deg2rad(0.5 * (grid.lat_min + grid.lat_max))
        km_per_deg_lon = km_per_deg_lat * np.cos(mean_lat)

        sigma_lat = self.smoothing_km / (km_per_deg_lat * grid.cell_deg)
        sigma_lon = self.smoothing_km / (max(km_per_deg_lon, 1e-6) * grid.cell_deg)
        return sigma_lat, sigma_lon

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

        counts = grid.bin_events(events).reshape(grid.shape)

        sigma_lat, sigma_lon = self._sigma_cells(grid)
        smoothed = gaussian_filter(counts, sigma=(sigma_lat, sigma_lon), mode="nearest").ravel()

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
            "model": "PoissonBaseline",
            "smoothing_km": self.smoothing_km,
            "min_magnitude": self.min_magnitude,
            "background_fraction": self.background_fraction,
            "time_dependent": False,
        }
