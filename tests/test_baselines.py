"""Baseline behaviour.

These check the contract every model must satisfy and a few properties that
follow from what the models claim to be. They are not accuracy tests — accuracy
is what the benchmark itself measures, and asserting a specific score here would
freeze the baselines against improvement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seismic_bench.baselines import (
    AdaptivePoissonBaseline,
    ETASBaseline,
    PoissonBaseline,
    validate_forecast,
)
from seismic_bench.baselines.base import Forecaster
from seismic_bench.bench.grid import CALIFORNIA
from seismic_bench.bench.split import Window, rolling_windows

MODELS = [PoissonBaseline, AdaptivePoissonBaseline, ETASBaseline]


@pytest.fixture(scope="module")
def window_and_training(sample_catalog):
    """A window containing the 2019 Ridgecrest sequence.

    Deliberately the hardest month in the sample: an M7.1 mainshock with a large
    aftershock sequence. A model that is fine on quiet months and breaks here is
    broken where it matters.
    """
    catalog = sample_catalog.loc[CALIFORNIA.contains(sample_catalog)].reset_index(drop=True)
    windows = rolling_windows(catalog, test_days=30, min_training_days=365 * 3)
    mainshock = pd.Timestamp("2019-07-06", tz="UTC")
    ridgecrest = [w for w in windows if w.test_start <= mainshock < w.test_end]
    window = ridgecrest[0] if ridgecrest else windows[-1]
    return window, window.training_events(catalog), catalog


@pytest.mark.parametrize("model_class", MODELS)
class TestForecastContract:
    def test_satisfies_the_protocol(self, model_class):
        assert isinstance(model_class(), Forecaster)

    def test_returns_one_rate_per_cell(self, model_class, window_and_training):
        window, training, _ = window_and_training
        rates = model_class().forecast(training, CALIFORNIA, window)
        assert rates.shape == (CALIFORNIA.n_cells,)

    def test_rates_are_finite_and_non_negative(self, model_class, window_and_training):
        window, training, _ = window_and_training
        rates = model_class().forecast(training, CALIFORNIA, window)
        assert np.all(np.isfinite(rates))
        assert np.all(rates >= 0)

    def test_passes_validation(self, model_class, window_and_training):
        window, training, _ = window_and_training
        model = model_class()
        rates = model.forecast(training, CALIFORNIA, window)
        validate_forecast(rates, CALIFORNIA, model.name)

    def test_forecasts_something(self, model_class, window_and_training):
        """A model predicting zero events everywhere is not a model."""
        window, training, _ = window_and_training
        assert model_class().forecast(training, CALIFORNIA, window).sum() > 0

    def test_handles_an_empty_training_set(self, model_class):
        """Should degrade to something uninformative, not crash or return zeros."""
        window = Window(
            train_start=pd.Timestamp("2020-01-01", tz="UTC"),
            train_end=pd.Timestamp("2020-02-01", tz="UTC"),
            test_start=pd.Timestamp("2020-02-01", tz="UTC"),
            test_end=pd.Timestamp("2020-03-01", tz="UTC"),
        )
        empty = pd.DataFrame(
            columns=["event_id", "time", "latitude", "longitude", "depth_km", "magnitude"]
        ).astype({"magnitude": float, "latitude": float, "longitude": float})

        rates = model_class().forecast(empty, CALIFORNIA, window)
        assert rates.shape == (CALIFORNIA.n_cells,)
        assert np.all(np.isfinite(rates))
        assert rates.sum() > 0

    def test_describe_is_populated(self, model_class):
        described = model_class().describe()
        assert isinstance(described, dict) and described
        assert "model" in described


class TestPoissonProperties:
    def test_scales_linearly_with_window_length(self, window_and_training):
        """Time-independent: twice the window, twice the expected count."""
        window, training, _ = window_and_training
        long_window = Window(
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_start + pd.Timedelta(days=60),
        )
        model = PoissonBaseline()
        short = model.forecast(training, CALIFORNIA, window).sum()
        long = model.forecast(training, CALIFORNIA, long_window).sum()
        assert long == pytest.approx(2.0 * short, rel=1e-9)

    def test_no_cell_is_exactly_zero(self, window_and_training):
        """The uniform background exists so no cell claims impossibility."""
        window, training, _ = window_and_training
        assert np.all(PoissonBaseline().forecast(training, CALIFORNIA, window) > 0)

    def test_concentrates_rate_near_past_events(self, window_and_training):
        """Smoothed seismicity should put more rate where events have occurred.

        The observed ratio is around 3, not orders of magnitude, and that is
        expected: a 50 km kernel on 0.2 degree (~20 km) cells deliberately
        spreads rate into the neighbourhood of every past event, so cells
        adjacent to activity carry substantial rate despite being "inactive"
        themselves. A much sharper contrast would mean the smoothing was not
        doing its job.
        """
        window, training, _ = window_and_training
        rates = PoissonBaseline().forecast(training, CALIFORNIA, window)

        counts = CALIFORNIA.bin_events(training.loc[training["magnitude"] >= 3.0])
        active = counts > 0
        assert active.sum() > 0
        assert rates[active].mean() > rates[~active].mean() * 2

    def test_rejects_an_invalid_background_fraction(self):
        with pytest.raises(ValueError, match="background_fraction"):
            PoissonBaseline(background_fraction=1.5)


def _tight_cluster(centre_lon: float, centre_lat: float, n: int = 40) -> pd.DataFrame:
    """A cluster of `n` events spread over roughly a hundred metres."""
    offsets = np.linspace(-0.0005, 0.0005, n)
    return pd.DataFrame(
        {
            "event_id": [f"c{i}" for i in range(n)],
            "time": pd.to_datetime(["2019-01-01T00:00:00Z"] * n, utc=True),
            "latitude": centre_lat + offsets,
            "longitude": centre_lon + offsets,
            "depth_km": 5.0,
            "magnitude": 3.5,
        }
    )


class TestAdaptivePoissonProperties:
    """The adaptive-bandwidth model, proposed by @nextbaires in #1.

    These assert the behaviour the bandwidth is supposed to produce, not that
    the arithmetic runs. A model that ignored its own nearest-neighbour
    distances and smoothed everything at one width would pass the contract tests
    above and fail every test here.
    """

    def test_bandwidth_narrows_where_the_catalog_is_dense(self, window_and_training):
        """The core claim: one bandwidth cannot serve all of California.

        The creeping section of the San Andreas produces events at rates far
        above the Mojave (per @nextbaires, #1), so its nearest-neighbour
        distances — and therefore its kernels — must come out smaller. This is
        the mechanism of Helmstetter, Kagan & Jackson (2007),
        https://doi.org/10.1785/gssrl.78.1.78.
        """
        _, training, _ = window_and_training
        lon = training["longitude"].to_numpy(dtype=float)
        lat = training["latitude"].to_numpy(dtype=float)
        bandwidth = AdaptivePoissonBaseline().bandwidths_km(lon, lat)

        creeping = (lat > 35.8) & (lat < 36.9) & (lon > -121.6) & (lon < -120.3)
        mojave = (lat > 34.2) & (lat < 35.2) & (lon > -118.3) & (lon < -116.5)
        assert creeping.sum() > 20 and mojave.sum() > 20

        assert np.median(bandwidth[creeping]) < np.median(bandwidth[mojave])

    def test_an_isolated_event_gets_a_wide_kernel(self):
        """The other half of the claim: sparse regions should not stay at the floor."""
        cluster = _tight_cluster(-120.0, 36.0)
        isolated = cluster.iloc[[0]].assign(longitude=-117.0, latitude=34.0)
        events = pd.concat([cluster, isolated], ignore_index=True)

        model = AdaptivePoissonBaseline()
        bandwidth = model.bandwidths_km(
            events["longitude"].to_numpy(dtype=float),
            events["latitude"].to_numpy(dtype=float),
        )
        assert bandwidth[-1] > 100.0 * bandwidth[:-1].max()

    def test_a_tight_cluster_is_held_at_the_bandwidth_floor(self):
        """Without the floor a near-coincident cluster is a delta function (#1).

        The floor is what stops that. Note that a few-kilometre floor is still
        well inside a 0.2 degree cell, so on this grid it bounds the kernel
        without visibly spreading it — widening the floor is what moves rate off
        the peak cell, and the second assertion is what demonstrates the floor is
        load-bearing at all.
        """
        events = _tight_cluster(-119.0, 35.0)
        window = Window(
            train_start=pd.Timestamp("2019-01-01", tz="UTC"),
            train_end=pd.Timestamp("2019-06-01", tz="UTC"),
            test_start=pd.Timestamp("2019-06-01", tz="UTC"),
            test_end=pd.Timestamp("2019-07-01", tz="UTC"),
        )

        model = AdaptivePoissonBaseline()
        bandwidth = model.bandwidths_km(
            events["longitude"].to_numpy(dtype=float),
            events["latitude"].to_numpy(dtype=float),
        )
        assert np.allclose(bandwidth, model.min_bandwidth_km)

        def peak_share(min_bandwidth_km: float) -> float:
            rates = AdaptivePoissonBaseline(min_bandwidth_km=min_bandwidth_km).forecast(
                events, CALIFORNIA, window
            )
            return float(rates.max() / rates.sum())

        assert peak_share(40.0) < 0.5 * peak_share(model.min_bandwidth_km)

    def test_concentrates_more_sharply_than_the_fixed_bandwidth_model(self, window_and_training):
        """The proposer's second check (#1), on the real catalog.

        Sharper where the events are: a larger share of the total rate lands in
        cells that have actually had an event. Correspondingly, cells far from
        any historical event are pulled up less than the 50 km kernel pulls them
        up.
        """
        window, training, _ = window_and_training
        fixed = PoissonBaseline().forecast(training, CALIFORNIA, window)
        adaptive = AdaptivePoissonBaseline().forecast(training, CALIFORNIA, window)

        counts = CALIFORNIA.bin_events(training.loc[training["magnitude"] >= 3.0])
        active = counts > 0
        assert adaptive[active].sum() / adaptive.sum() > fixed[active].sum() / fixed.sum()

        cell_lon, cell_lat = CALIFORNIA.cell_centers()
        event_lon = training["longitude"].to_numpy(dtype=float)
        event_lat = training["latitude"].to_numpy(dtype=float)
        km_per_deg_lon = 111.19 * np.cos(np.deg2rad(cell_lat))
        distance = np.sqrt(
            ((cell_lon[:, None] - event_lon[None, :]) * km_per_deg_lon[:, None]) ** 2
            + ((cell_lat[:, None] - event_lat[None, :]) * 111.19) ** 2
        ).min(axis=1)
        far = distance > 100.0
        assert far.sum() > 100
        assert adaptive[far].mean() < fixed[far].mean()

    def test_forecasts_the_same_total_as_the_fixed_bandwidth_model(self, window_and_training):
        """Only the spatial distribution changes, which is why the N-test should not.

        If this drifts, a difference in the S-test between the two models stops
        being attributable to the bandwidth alone.
        """
        window, training, _ = window_and_training
        fixed = PoissonBaseline().forecast(training, CALIFORNIA, window).sum()
        adaptive = AdaptivePoissonBaseline().forecast(training, CALIFORNIA, window).sum()
        assert adaptive == pytest.approx(fixed, rel=1e-9)

    def test_weight_does_not_depend_on_where_a_cluster_sits_inside_a_cell(self):
        """Two identical clusters, one on a cell centre and one on a cell corner.

        With bandwidths smaller than a cell — which is most of them on this
        catalog — sampling the kernel at cell centres instead of integrating it
        over the cell would hand the centred cluster orders of magnitude more
        weight than the one straddling a corner. That is a property of the grid,
        not of the seismicity, so the two must come out with the same rate.
        """
        half = 0.5 * CALIFORNIA.cell_deg
        centre_lon, centre_lat = -119.0 + half, 35.0 + half
        corner_lon, corner_lat = -118.0, 36.0
        events = pd.concat(
            [
                _tight_cluster(centre_lon, centre_lat),
                _tight_cluster(corner_lon, corner_lat),
            ],
            ignore_index=True,
        )
        window = Window(
            train_start=pd.Timestamp("2019-01-01", tz="UTC"),
            train_end=pd.Timestamp("2019-06-01", tz="UTC"),
            test_start=pd.Timestamp("2019-06-01", tz="UTC"),
            test_end=pd.Timestamp("2019-07-01", tz="UTC"),
        )
        rates = AdaptivePoissonBaseline().forecast(events, CALIFORNIA, window)

        def neighbourhood(lon: float, lat: float) -> float:
            cell_lon, cell_lat = CALIFORNIA.cell_centers()
            near = (np.abs(cell_lon - lon) < CALIFORNIA.cell_deg) & (
                np.abs(cell_lat - lat) < CALIFORNIA.cell_deg
            )
            return float(rates[near].sum())

        centred = neighbourhood(centre_lon, centre_lat)
        cornered = neighbourhood(corner_lon, corner_lat)
        assert cornered == pytest.approx(centred, rel=0.05)

    def test_rejects_an_invalid_neighbour_count(self):
        with pytest.raises(ValueError, match="n_neighbours"):
            AdaptivePoissonBaseline(n_neighbours=0)

    def test_rejects_a_non_positive_bandwidth_floor(self):
        with pytest.raises(ValueError, match="min_bandwidth_km"):
            AdaptivePoissonBaseline(min_bandwidth_km=0.0)

    def test_declares_that_the_bandwidth_is_not_fit(self):
        """Same honesty as the ETAS parameters — assert it stays true."""
        assert AdaptivePoissonBaseline().describe()["bandwidth_fit"] is False


class TestETASProperties:
    def test_forecasts_more_than_poisson_after_a_mainshock(self, sample_catalog):
        """The whole point of a time-dependent model.

        Note the window starts *two days after* Ridgecrest, not in the month
        containing it. In the containing window the mainshock falls on the test
        side of the boundary, so ETAS cannot see it and correctly forecasts no
        elevated rate — which is the lookahead guard doing its job, not a
        modelling failure. The elevated forecast is only legitimate once the
        mainshock is genuinely in the past.
        """
        catalog = sample_catalog.loc[CALIFORNIA.contains(sample_catalog)].reset_index(drop=True)
        test_start = pd.Timestamp("2019-07-08", tz="UTC")
        window = Window(
            train_start=catalog["time"].min(),
            train_end=test_start,
            test_start=test_start,
            test_end=test_start + pd.Timedelta(days=30),
        )
        training = window.training_events(catalog)
        assert training["magnitude"].max() >= 7.0, "training should contain the mainshock"

        poisson_total = PoissonBaseline().forecast(training, CALIFORNIA, window).sum()
        etas_total = ETASBaseline().forecast(training, CALIFORNIA, window).sum()
        assert etas_total > poisson_total

    def test_triggering_decays_with_time_since_the_mainshock(self, sample_catalog):
        """Two windows, same model, later one further from the sequence."""
        catalog = sample_catalog.loc[CALIFORNIA.contains(sample_catalog)].reset_index(drop=True)
        model = ETASBaseline()

        def total_for(start: str) -> float:
            test_start = pd.Timestamp(start, tz="UTC")
            window = Window(
                train_start=catalog["time"].min(),
                train_end=test_start,
                test_start=test_start,
                test_end=test_start + pd.Timedelta(days=30),
            )
            return model.forecast(window.training_events(catalog), CALIFORNIA, window).sum()

        # Ridgecrest mainshock: 2019-07-06.
        just_after = total_for("2019-07-08")
        months_later = total_for("2019-11-08")
        assert just_after > months_later

    def test_rejects_a_divergent_omori_exponent(self):
        with pytest.raises(ValueError, match="p must be > 1"):
            ETASBaseline(p=0.9)

    def test_rejects_an_unnormalisable_spatial_kernel(self):
        with pytest.raises(ValueError, match="q must be > 1"):
            ETASBaseline(q=1.0)

    def test_declares_that_it_is_not_fit(self):
        """The honesty is part of the artefact — assert it stays true."""
        assert ETASBaseline().describe()["parameters_fit"] is False


class TestValidateForecast:
    def test_rejects_a_wrong_shape(self):
        with pytest.raises(ValueError, match="expected"):
            validate_forecast(np.ones(5), CALIFORNIA, "toy")

    def test_rejects_negative_rates(self):
        rates = np.ones(CALIFORNIA.n_cells)
        rates[0] = -1.0
        with pytest.raises(ValueError, match="negative"):
            validate_forecast(rates, CALIFORNIA, "toy")

    def test_rejects_nan(self):
        rates = np.ones(CALIFORNIA.n_cells)
        rates[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            validate_forecast(rates, CALIFORNIA, "toy")

    def test_error_names_the_model(self):
        with pytest.raises(ValueError, match="my-model"):
            validate_forecast(np.ones(3), CALIFORNIA, "my-model")
