"""Scoring correctness.

The log-likelihood is checked against values computed by hand, not against the
implementation's own output, so a sign error or a dropped factorial term cannot
pass by agreeing with itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from seismic_bench.bench import metrics


class TestLogLikelihood:
    def test_matches_hand_computed_value(self):
        """Poisson LL for a two-cell case, worked out independently.

        cell 0: lambda=2, n=1 -> -2 + 1*ln(2) - ln(1!) = -2 + 0.693147 - 0
        cell 1: lambda=1, n=0 -> -1 + 0        - ln(0!) = -1
        """
        expected = (-2.0 + math.log(2.0)) + (-1.0)
        got = metrics.log_likelihood(np.array([2.0, 1.0]), np.array([1.0, 0.0]))
        assert got == pytest.approx(expected, rel=1e-12)

    def test_factorial_term_is_present(self):
        """lambda=1, n=3 -> -1 + 3*ln(1) - ln(3!) = -1 - ln(6).

        Omitting the log-factorial is the classic Poisson-likelihood bug. It does
        not change model *rankings* when counts are identical, which is exactly
        why it survives undetected.
        """
        expected = -1.0 - math.log(6.0)
        got = metrics.log_likelihood(np.array([1.0]), np.array([3.0]))
        assert got == pytest.approx(expected, rel=1e-12)

    def test_perfect_forecast_beats_a_wrong_one(self):
        observed = np.array([5.0, 0.0, 2.0])
        good = metrics.log_likelihood(np.array([5.0, 0.1, 2.0]), observed)
        bad = metrics.log_likelihood(np.array([0.1, 5.0, 0.1]), observed)
        assert good > bad

    def test_zero_rates_are_floored_not_infinite(self):
        """A zero rate where an event occurred must not return -inf."""
        got = metrics.log_likelihood(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
        assert np.isfinite(got)

    def test_flooring_is_reported(self):
        scores = metrics.score(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]))
        assert scores.n_floored_cells == 2

    def test_negative_rates_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            metrics.log_likelihood(np.array([-1.0]), np.array([0.0]))

    def test_nonfinite_rates_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            metrics.log_likelihood(np.array([np.nan]), np.array([0.0]))

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="shape"):
            metrics.log_likelihood(np.array([1.0, 2.0]), np.array([1.0]))


class TestNTest:
    def test_well_calibrated_total_passes(self):
        """Forecast 100, observe 100: both tails comfortably non-extreme."""
        forecast = np.full(100, 1.0)
        observed = np.zeros(100)
        observed[:100] = 1.0
        delta1, delta2 = metrics.n_test(forecast, observed)
        assert delta1 > 0.025 and delta2 > 0.025

    def test_gross_under_forecast_fails_delta1(self):
        """Forecast ~1 event, observe 50. delta1 = P(N >= 50) is tiny."""
        delta1, _ = metrics.n_test(np.full(10, 0.1), np.full(10, 5.0))
        assert delta1 < 0.025

    def test_gross_over_forecast_fails_delta2(self):
        """Forecast 1000, observe 1. delta2 = P(N <= 1) is tiny."""
        observed = np.zeros(10)
        observed[0] = 1.0
        _, delta2 = metrics.n_test(np.full(10, 100.0), observed)
        assert delta2 < 0.025

    def test_deltas_are_probabilities(self):
        delta1, delta2 = metrics.n_test(np.full(20, 0.5), np.full(20, 0.5))
        assert 0.0 <= delta1 <= 1.0
        assert 0.0 <= delta2 <= 1.0


class TestSTest:
    def test_correct_spatial_pattern_passes(self, rng):
        """Events drawn from the forecast's own spatial distribution."""
        rates = rng.uniform(0.1, 2.0, size=200)
        probabilities = rates / rates.sum()
        observed = rng.multinomial(300, probabilities).astype(float)
        assert metrics.s_test(rates, observed, n_simulations=2000) >= 0.05

    def test_wrong_spatial_pattern_fails(self, rng):
        """All events land where the forecast said they would not."""
        rates = np.concatenate([np.full(100, 5.0), np.full(100, 0.01)])
        observed = np.zeros(200)
        observed[100:] = 3.0  # every event in the low-rate half
        assert metrics.s_test(rates, observed, n_simulations=2000) < 0.05

    def test_is_insensitive_to_total_rate(self, rng):
        """Scaling the whole forecast changes the N-test, not the S-test.

        This is the property that makes the two tests worth running separately —
        if scaling moved gamma, the S-test would just be a second N-test.
        """
        rates = rng.uniform(0.1, 2.0, size=150)
        observed = rng.multinomial(200, rates / rates.sum()).astype(float)

        base = metrics.s_test(rates, observed, n_simulations=4000, seed=7)
        scaled = metrics.s_test(rates * 10.0, observed, n_simulations=4000, seed=7)
        assert base == pytest.approx(scaled, abs=1e-9)

    def test_no_observed_events_is_not_a_pass(self):
        """An empty window is untestable spatially, and must not read as a pass."""
        assert math.isnan(metrics.s_test(np.full(10, 1.0), np.zeros(10)))

    def test_is_reproducible_under_a_seed(self, rng):
        rates = rng.uniform(0.1, 2.0, size=50)
        observed = rng.multinomial(60, rates / rates.sum()).astype(float)
        a = metrics.s_test(rates, observed, n_simulations=1000, seed=42)
        b = metrics.s_test(rates, observed, n_simulations=1000, seed=42)
        assert a == b


class TestBrier:
    def test_perfect_certainty_scores_near_zero(self):
        """High rate where events occurred, ~0 where they did not."""
        forecast = np.array([50.0, 1e-9, 50.0])
        observed = np.array([3.0, 0.0, 1.0])
        assert metrics.brier_score(forecast, observed) < 1e-6

    def test_confidently_wrong_scores_near_one(self):
        forecast = np.array([50.0, 1e-9])
        observed = np.array([0.0, 4.0])
        assert metrics.brier_score(forecast, observed) > 0.9

    def test_bounded_in_unit_interval(self, rng):
        forecast = rng.uniform(0, 5, size=100)
        observed = rng.poisson(1.0, size=100).astype(float)
        assert 0.0 <= metrics.brier_score(forecast, observed) <= 1.0


class TestScore:
    def test_bundles_every_metric(self, rng):
        rates = rng.uniform(0.1, 2.0, size=80)
        observed = rng.multinomial(100, rates / rates.sum()).astype(float)

        scores = metrics.score(rates, observed, n_simulations=1000)
        assert scores.n_observed == 100
        assert scores.n_forecast == pytest.approx(rates.sum())
        assert np.isfinite(scores.log_likelihood)
        assert 0.0 <= scores.brier <= 1.0
        assert set(scores.as_dict()) == {
            "log_likelihood",
            "n_test_delta1",
            "n_test_delta2",
            "s_test_gamma",
            "brier",
            "n_forecast",
            "n_observed",
            "n_floored_cells",
        }
