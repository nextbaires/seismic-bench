"""The lookahead guard.

This is the most important test file in the repository. Every other test checks
that something computes correctly; these check that the benchmark cannot be
cheated, which is the only reason to trust any number it produces.

If one of these fails, no result from this repository means anything until it is
fixed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from seismic_bench.bench.split import Window, rolling_windows


def _catalog(times: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(len(times))],
            "time": pd.to_datetime(times, utc=True),
            "latitude": 35.0,
            "longitude": -119.0,
            "depth_km": 5.0,
            "magnitude": 4.0,
            "mag_type": "mw",
            "source": "test",
        }
    )


class TestNoLookahead:
    def test_training_events_never_reach_the_forecast_window(self, sample_catalog):
        """The core guarantee, on real data, on every window."""
        windows = rolling_windows(sample_catalog, test_days=30, min_training_days=365 * 3)
        assert windows, "sample catalog should produce windows"

        for window in windows:
            training = window.training_events(sample_catalog)
            if training.empty:
                continue
            assert training["time"].max() < window.test_start, (
                f"training data reaches {training['time'].max()} but the forecast "
                f"window opens at {window.test_start}"
            )

    def test_boundary_event_belongs_to_test_not_training(self):
        """An event at the exact instant the window opens must not train the model.

        Half-open intervals on both sides. Getting this wrong leaks exactly one
        event per window — invisible in aggregate, and enough to flatter a
        short-term model that keys on the most recent event.
        """
        boundary = "2020-06-01T00:00:00Z"
        catalog = _catalog(["2020-05-31T23:59:59Z", boundary, "2020-06-01T00:00:01Z"])

        window = Window(
            train_start=pd.Timestamp("2020-01-01", tz="UTC"),
            train_end=pd.Timestamp(boundary),
            test_start=pd.Timestamp(boundary),
            test_end=pd.Timestamp("2020-07-01", tz="UTC"),
        )

        training = window.training_events(catalog)
        test = window.test_events(catalog)

        assert list(training["event_id"]) == ["e0"]
        assert list(test["event_id"]) == ["e1", "e2"]
        assert pd.Timestamp(boundary) not in set(training["time"])

    def test_constructing_an_overlapping_window_raises(self):
        """A window whose training period runs past the forecast start is rejected."""
        with pytest.raises(ValueError, match="lookahead leak"):
            Window(
                train_start=pd.Timestamp("2020-01-01", tz="UTC"),
                train_end=pd.Timestamp("2020-07-01", tz="UTC"),  # past test_start
                test_start=pd.Timestamp("2020-06-01", tz="UTC"),
                test_end=pd.Timestamp("2020-07-01", tz="UTC"),
            )

    def test_every_generated_window_is_internally_consistent(self, sample_catalog):
        for window in rolling_windows(sample_catalog, test_days=30, min_training_days=365 * 3):
            assert window.train_start < window.train_end
            assert window.train_end <= window.test_start
            assert window.test_start < window.test_end


class TestWindowGeneration:
    def test_windows_do_not_overlap(self, sample_catalog):
        """Per-window scores are summed, so any overlap double-counts events."""
        windows = rolling_windows(sample_catalog, test_days=30, min_training_days=365 * 3)
        for earlier, later in zip(windows, windows[1:], strict=False):
            assert earlier.test_end <= later.test_start

    def test_windows_are_contiguous(self, sample_catalog):
        """No gaps either — a gap silently drops events from evaluation."""
        windows = rolling_windows(sample_catalog, test_days=30, min_training_days=365 * 3)
        for earlier, later in zip(windows, windows[1:], strict=False):
            assert earlier.test_end == later.test_start

    def test_expanding_training_grows(self, sample_catalog):
        windows = rolling_windows(sample_catalog, test_days=30, min_training_days=365 * 3)
        starts = {w.train_start for w in windows}
        assert len(starts) == 1, "expanding windows share one start"
        assert all(
            earlier.train_end < later.train_end
            for earlier, later in zip(windows, windows[1:], strict=False)
        )

    def test_sliding_training_has_constant_length(self, sample_catalog):
        windows = rolling_windows(
            sample_catalog,
            test_days=30,
            min_training_days=365 * 3,
            expanding=False,
            train_days=365,
        )
        lengths = {round(w.training_days, 6) for w in windows}
        assert len(lengths) == 1, f"sliding windows should be one length, got {lengths}"

    def test_sliding_requires_train_days(self, sample_catalog):
        with pytest.raises(ValueError, match="train_days is required"):
            rolling_windows(sample_catalog, expanding=False)

    def test_empty_catalog_yields_no_windows(self):
        assert rolling_windows(_catalog([])) == []

    def test_catalog_shorter_than_training_yields_no_windows(self):
        catalog = _catalog(["2020-01-01T00:00:00Z", "2020-02-01T00:00:00Z"])
        assert rolling_windows(catalog, min_training_days=365 * 3) == []

    def test_duration_days_matches_test_days(self, sample_catalog):
        for window in rolling_windows(sample_catalog, test_days=30, min_training_days=365 * 3):
            assert window.duration_days == pytest.approx(30.0)
