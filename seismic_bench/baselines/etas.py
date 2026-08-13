"""ETAS — the standard time-dependent reference.

The Epidemic-Type Aftershock Sequence model (Ogata 1988,
https://doi.org/10.1080/01621459.1988.10478560) treats seismicity as a background
rate plus triggering: every earthquake raises the probability of further
earthquakes nearby and soon after, decaying as a power law in both time
(Omori-Utsu) and distance, with the boost scaling exponentially in magnitude.

    lambda(t, x) = mu(x) + sum over past events i of
                   K * 10^(alpha * (m_i - Mc))
                   * (t - t_i + c)^(-p)
                   * f(|x - x_i| ; d_i, q)

Expected counts per cell are obtained by integrating this over the forecast
window analytically in time, which avoids the discretisation error of stepping
through it.

What this implementation is not
-------------------------------
**The parameters are fixed literature values, not fit to the catalog.** A real
ETAS application estimates (K, alpha, c, p, d0, gamma, q) by maximum likelihood
on the target region, and the estimates vary substantially between regions and
between tectonic settings. The defaults here are representative Southern
California values.

This is a deliberate, and clearly suboptimal, choice. Fitting ETAS properly
inside a rolling-origin benchmark means refitting on each window's training data
only — doable, but a meaningful piece of work with its own failure modes
(convergence, parameter drift, the fit itself leaking if written carelessly).
Shipping an unfit ETAS that is honest about being unfit is better than shipping a
fit one whose fitting procedure has not been reviewed.

**So this baseline is beatable, and beating it is not yet an achievement.** The
first genuinely valuable contribution to this repository is probably per-window
maximum-likelihood estimation of these parameters. If you know how to do that
properly, please say so — you do not need to write the code.

The spatial kernel is also evaluated at cell centres rather than integrated over
each cell, which under-counts the cell containing the mainshock when the kernel
is narrow relative to the cell. Correcting it matters most for large events and
fine grids.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..bench.grid import Grid
from ..bench.split import Window
from .poisson import PoissonBaseline

KM_PER_DEG_LAT = 111.19


class ETASBaseline:
    """Time-dependent aftershock-triggering forecast with fixed parameters.

    Parameters
    ----------
    productivity_k, alpha
        Triggering strength. `alpha` controls how sharply productivity rises
        with magnitude; ~0.8 in base-10 form is typical for California.
    c_days, p
        Omori-Utsu time decay. `c` regularises the singularity at t = t_i;
        `p` slightly above 1 is near-universal.
    d0_km, gamma, q
        Spatial kernel. Aftershock zone size scales with magnitude through
        `gamma`; `q` sets the power-law tail.
    reference_magnitude
        The Mc that productivity and kernel size are measured relative to. Must
        match the completeness magnitude the catalog is filtered at, or the
        productivity constant means something different from what it says.
    max_history_days
        Only events this recent contribute triggering. With p ~ 1.1 the
        contribution from a year-old event is negligible, and the cutoff keeps
        the calculation from scaling with the whole catalog.
    background_weight
        Share of the long-run rate assigned to the background term, the rest
        being explained by triggering. ~0.3 is a common empirical split for
        California.
    """

    def __init__(
        self,
        *,
        productivity_k: float = 0.008,
        alpha: float = 0.8,
        c_days: float = 0.02,
        p: float = 1.1,
        d0_km: float = 0.5,
        gamma: float = 0.5,
        q: float = 1.5,
        reference_magnitude: float = 3.0,
        max_history_days: float = 365.0,
        background_weight: float = 0.3,
    ) -> None:
        if p <= 1.0:
            # p <= 1 makes the Omori integral diverge over an infinite window.
            # It stays finite over a bounded one, but a model whose expected
            # aftershock count diverges with observation time is not one to
            # silently accept.
            raise ValueError("p must be > 1 for the Omori integral to converge")
        if q <= 1.0:
            raise ValueError("q must be > 1 for the spatial kernel to normalise")
        if not 0.0 <= background_weight <= 1.0:
            raise ValueError("background_weight must be in [0, 1]")

        self.productivity_k = float(productivity_k)
        self.alpha = float(alpha)
        self.c_days = float(c_days)
        self.p = float(p)
        self.d0_km = float(d0_km)
        self.gamma = float(gamma)
        self.q = float(q)
        self.reference_magnitude = float(reference_magnitude)
        self.max_history_days = float(max_history_days)
        self.background_weight = float(background_weight)
        self.name = "etas"

        self._background_model = PoissonBaseline(
            smoothing_km=50.0,
            min_magnitude=reference_magnitude,
            background_fraction=0.01,
        )

    # -- pieces of the model -------------------------------------------------

    def _omori_integral(self, t_since_i_start: np.ndarray, t_since_i_end: np.ndarray) -> np.ndarray:
        """Time integral of (t - t_i + c)^-p across the forecast window.

        Analytic, so window length does not introduce discretisation error:

            integral = [(T1 - t_i + c)^(1-p) - (T0 - t_i + c)^(1-p)] / (1 - p)
        """
        one_minus_p = 1.0 - self.p
        start = np.maximum(t_since_i_start, 0.0) + self.c_days
        end = np.maximum(t_since_i_end, 0.0) + self.c_days
        return (end**one_minus_p - start**one_minus_p) / one_minus_p

    def _spatial_kernel(self, distance_km: np.ndarray, d_km: np.ndarray) -> np.ndarray:
        """Normalised 2-D power law: (q-1)/(pi d^2) * (1 + r^2/d^2)^-q.

        Integrates to 1 over the plane, so the productivity term alone controls
        how many aftershocks are expected and the kernel only decides where.
        """
        d2 = d_km**2
        return ((self.q - 1.0) / (np.pi * d2)) * (1.0 + (distance_km**2) / d2) ** (-self.q)

    # -- interface -----------------------------------------------------------

    def forecast(self, training_events: pd.DataFrame, grid: Grid, window: Window) -> np.ndarray:
        background = self._background_model.forecast(training_events, grid, window)

        events = training_events
        if not events.empty:
            events = events.loc[events["magnitude"] >= self.reference_magnitude]

        if events.empty:
            return background

        # Restrict to events recent enough to still be triggering.
        cutoff = window.test_start - pd.Timedelta(days=self.max_history_days)
        recent = events.loc[events["time"] >= cutoff]
        if recent.empty:
            return background

        cell_lon, cell_lat = grid.cell_centers()
        cell_areas = grid.cell_areas_km2()

        event_lon = recent["longitude"].to_numpy(dtype=float)
        event_lat = recent["latitude"].to_numpy(dtype=float)
        event_mag = recent["magnitude"].to_numpy(dtype=float)

        # Days from each event to the window's start and end. Computed through
        # pandas rather than numpy: a tz-aware datetime column becomes an object
        # array under `.to_numpy()`, and subtracting it from a Timestamp raises.
        days_to_start = (
            (window.test_start - recent["time"]).dt.total_seconds().to_numpy() / 86400.0
        )
        days_to_end = (
            (window.test_end - recent["time"]).dt.total_seconds().to_numpy() / 86400.0
        )

        excess = event_mag - self.reference_magnitude
        productivity = self.productivity_k * (10.0**(self.alpha * excess))
        time_factor = self._omori_integral(days_to_start, days_to_end)
        kernel_width = self.d0_km * (10.0**(self.gamma * excess))

        # Expected aftershocks from each event over the window, before spatial
        # distribution. Events contributing effectively nothing are dropped
        # rather than carried through the pairwise distance calculation.
        strength = productivity * time_factor
        keep = strength > 1e-12
        if not np.any(keep):
            return background

        event_lon, event_lat = event_lon[keep], event_lat[keep]
        strength, kernel_width = strength[keep], kernel_width[keep]

        km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.deg2rad(np.clip(event_lat, -89.9, 89.9)))

        triggered = np.zeros(grid.n_cells, dtype=float)
        # Chunked so memory stays bounded on long catalogs — the pairwise array
        # is (chunk x cells), not (events x cells).
        chunk = max(1, int(2_000_000 // max(grid.n_cells, 1)))
        for lo in range(0, len(strength), chunk):
            hi = lo + chunk
            dlon_km = (cell_lon[None, :] - event_lon[lo:hi, None]) * km_per_deg_lon[lo:hi, None]
            dlat_km = (cell_lat[None, :] - event_lat[lo:hi, None]) * KM_PER_DEG_LAT
            distance = np.sqrt(dlon_km**2 + dlat_km**2)

            density = self._spatial_kernel(distance, kernel_width[lo:hi, None])
            triggered += np.sum(strength[lo:hi, None] * density * cell_areas[None, :], axis=0)

        return self.background_weight * background + triggered

    def describe(self) -> dict[str, object]:
        return {
            "model": "ETASBaseline",
            "productivity_k": self.productivity_k,
            "alpha": self.alpha,
            "c_days": self.c_days,
            "p": self.p,
            "d0_km": self.d0_km,
            "gamma": self.gamma,
            "q": self.q,
            "reference_magnitude": self.reference_magnitude,
            "max_history_days": self.max_history_days,
            "background_weight": self.background_weight,
            "parameters_fit": False,
            "parameter_source": (
                "fixed literature values, Southern California; not fit to this catalog"
            ),
            "time_dependent": True,
        }
