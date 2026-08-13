"""What a forecast model is.

A model is anything with a `forecast` method taking the training events, the
grid, and the window, and returning one non-negative rate per cell — the
*expected number of events in that cell over that window*, not a rate per day
and not a probability.

The signature is the whole interface deliberately. A model receives exactly the
events it is allowed to see (`Window.training_events` guarantees that) and gets
no handle on the catalog, the filesystem, or the future. Anything a model needs
beyond the training events is a parameter it declares in its own constructor,
where a reviewer can see it.

To submit a model, implement this and open a pull request adding it under
`seismic_bench/baselines/` or a `models/` directory of your own. If you are not
a programmer, describe it in an issue instead — see the expert loop in the
README.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..bench.grid import Grid
from ..bench.split import Window


@runtime_checkable
class Forecaster(Protocol):
    """The interface every model implements."""

    #: Short identifier used in the leaderboard. Must be unique.
    name: str

    def forecast(self, training_events: pd.DataFrame, grid: Grid, window: Window) -> np.ndarray:
        """Expected event counts per cell for `window`.

        Parameters
        ----------
        training_events
            Catalog rows strictly before `window.test_start`. Already filtered;
            do not attempt to widen it.
        grid
            The spatial binning. Return an array of length `grid.n_cells` in the
            same flattened lat-major order as `grid.cell_centers()`.
        window
            The forecast window. `window.duration_days` is the length to scale a
            per-day rate by.

        Returns
        -------
        np.ndarray
            Non-negative, finite, shape `(grid.n_cells,)`.
        """
        ...

    def describe(self) -> dict[str, object]:
        """Parameters, for the leaderboard record.

        Whatever a reader would need to reproduce the run. A model whose
        `describe` returns `{}` is one nobody can check.
        """
        ...


def validate_forecast(rates: np.ndarray, grid: Grid, model_name: str) -> np.ndarray:
    """Check a model's output before it reaches the scorer.

    Catching this here rather than in the metrics means the error names the model
    that produced it, which matters when a benchmark run covers a dozen windows
    and one of them fails.
    """
    rates = np.asarray(rates, dtype=float)

    if rates.shape != (grid.n_cells,):
        raise ValueError(
            f"{model_name} returned shape {rates.shape}, expected ({grid.n_cells},). "
            "Rates must be flattened lat-major, matching grid.cell_centers()."
        )
    if not np.all(np.isfinite(rates)):
        n_bad = int(np.sum(~np.isfinite(rates)))
        raise ValueError(f"{model_name} returned {n_bad} non-finite rate(s)")
    if np.any(rates < 0):
        n_bad = int(np.sum(rates < 0))
        raise ValueError(f"{model_name} returned {n_bad} negative rate(s)")

    return rates
