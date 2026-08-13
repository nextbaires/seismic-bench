"""Rolling-origin evaluation windows, and the guarantee that nothing leaks.

This is the part of the benchmark most worth being paranoid about. Every
optimistic earthquake-forecasting result has, somewhere in it, a model that saw
something it would not have had at the time. The leak is rarely dramatic: a
declustering step applied to the whole catalog, a completeness magnitude
estimated over the full period, a smoothing kernel fit on all the data and then
"evaluated" on a subset of it.

So the split is not a helper here, it is the contract. A model is handed a
`Window`, and the only catalog access it gets is `training_events`, which is
constructed to end strictly before the forecast window opens. There is no
argument a model can pass to see past that boundary, and `tests/test_split.py`
asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class Window:
    """One forecast window and the data legitimately available before it.

    `train_start`/`train_end` and `test_start`/`test_end` are both half-open:
    `[start, end)`. `train_end == test_start`, so an event exactly on the
    boundary belongs to the test window — the side that does *not* help the
    model.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.train_end > self.test_start:
            raise ValueError(
                f"training window ends at {self.train_end}, after the forecast window "
                f"opens at {self.test_start} — this is a lookahead leak"
            )
        if self.train_start >= self.train_end:
            raise ValueError("training window must be non-empty")
        if self.test_start >= self.test_end:
            raise ValueError("forecast window must be non-empty")

    @property
    def duration_days(self) -> float:
        """Length of the forecast window. Rates are per-window, so models need it."""
        return (self.test_end - self.test_start).total_seconds() / 86400.0

    @property
    def training_days(self) -> float:
        return (self.train_end - self.train_start).total_seconds() / 86400.0

    def training_events(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """Everything a model is allowed to see for this window. Nothing else.

        Half-open on both ends, so an event at exactly `train_end` — which is to
        say, the first instant of the forecast window — is excluded.
        """
        mask = (catalog["time"] >= self.train_start) & (catalog["time"] < self.train_end)
        return catalog.loc[mask].reset_index(drop=True)

    def test_events(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """The truth for this window. For scoring only — never pass this to a model."""
        mask = (catalog["time"] >= self.test_start) & (catalog["time"] < self.test_end)
        return catalog.loc[mask].reset_index(drop=True)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Window(train {self.train_start:%Y-%m-%d}..{self.train_end:%Y-%m-%d}, "
            f"test {self.test_start:%Y-%m-%d}..{self.test_end:%Y-%m-%d})"
        )


def rolling_windows(
    catalog: pd.DataFrame,
    *,
    test_days: int = 30,
    min_training_days: int = 365 * 3,
    expanding: bool = True,
    train_days: int | None = None,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> list[Window]:
    """Generate consecutive, non-overlapping forecast windows.

    Parameters
    ----------
    test_days
        Length of each forecast window. Thirty days is the CSEP short-term
        convention and is the default here.
    min_training_days
        How much history must accumulate before the first forecast. Three years
        is enough for a rate estimate to be worth anything and short enough to
        leave most of a decade-long catalog available for testing.
    expanding
        `True` grows the training window from the catalog start (each forecast
        uses all prior history). `False` uses a fixed-length sliding window of
        `train_days`, which is the right choice for testing whether a model is
        tracking non-stationarity rather than averaging it away.
    train_days
        Length of the sliding window. Required when `expanding=False`.

    Windows do not overlap, so per-window scores can be summed without
    double-counting an event.
    """
    if catalog.empty:
        return []
    if not expanding and train_days is None:
        raise ValueError("train_days is required when expanding=False")

    times = pd.to_datetime(catalog["time"], utc=True)
    catalog_start = pd.Timestamp(start, tz="UTC") if start is not None else times.min()
    catalog_end = pd.Timestamp(end, tz="UTC") if end is not None else times.max()

    test_delta = timedelta(days=test_days)
    first_test_start = catalog_start + timedelta(days=min_training_days)

    windows: list[Window] = []
    test_start = first_test_start
    while test_start + test_delta <= catalog_end:
        test_end = test_start + test_delta
        train_end = test_start
        if expanding:
            train_start = catalog_start
        else:
            train_start = max(catalog_start, train_end - timedelta(days=train_days))
        windows.append(
            Window(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        test_start = test_end

    return windows
