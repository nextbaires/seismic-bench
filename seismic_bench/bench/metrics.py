"""Scoring.

The metrics are the CSEP consistency tests, because using the tests the field
already argues about is worth more than inventing cleaner ones nobody trusts.
Each answers a different question, and a model can pass one while failing
another — which is the point of running all of them.

- **Log-likelihood** — one number for ranking. Rewards getting rate *and*
  location right, and is dominated by whichever the model gets badly wrong.
- **N-test** — did the model forecast the right *number* of events, ignoring
  where? A model can be spatially excellent and still fail this by being
  uniformly too high.
- **S-test** — was the *spatial pattern* right, controlling for the total? This
  is the one that separates "knows where earthquakes happen" from "knows the
  regional rate", and a smoothed-seismicity model that fails the N-test can pass
  it comfortably.
- **Brier score** — a proper score for the binary "any event in this cell"
  question. Included because it is interpretable to people who do not read
  likelihoods, and because it is much less sensitive to a single cell being
  badly wrong than the log-likelihood is.

Zero rates
----------
A forecast of exactly zero in a cell where an event occurs makes the
log-likelihood negative infinity — one such cell destroys the score regardless of
everything else. That is arguably correct (the model claimed the impossible and
was wrong) but it makes scores incomparable in practice. Rates are therefore
floored at `RATE_FLOOR` before scoring, and `forecast_diagnostics` reports how
many cells the floor touched. A model relying on the floor is a model with a
problem, and the diagnostic is there so that shows up in review rather than
hiding inside a finite number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson

#: Rates below this are raised to it before scoring. Equivalent to a prior of
#: about one event per cell per 100,000 windows — small enough not to flatter a
#: model, large enough to keep the score finite.
RATE_FLOOR = 1e-5

#: Simulations used for the S-test null distribution. 10,000 gives a gamma
#: resolution of 1e-4, comfortably finer than any threshold anyone tests at.
N_SIMULATIONS = 10_000


@dataclass(frozen=True)
class Scores:
    log_likelihood: float
    n_test_delta1: float
    n_test_delta2: float
    s_test_gamma: float
    brier: float
    n_forecast: float
    n_observed: int
    n_floored_cells: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _prepare(forecast: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    rates = np.asarray(forecast, dtype=float)
    counts = np.asarray(observed, dtype=float)

    if rates.shape != counts.shape:
        raise ValueError(f"forecast shape {rates.shape} != observed shape {counts.shape}")
    if not np.all(np.isfinite(rates)):
        raise ValueError("forecast contains non-finite rates")
    if np.any(rates < 0):
        raise ValueError("forecast contains negative rates")

    n_floored = int(np.sum(rates < RATE_FLOOR))
    return np.maximum(rates, RATE_FLOOR), counts, n_floored


def log_likelihood(forecast: np.ndarray, observed: np.ndarray) -> float:
    """Joint Poisson log-likelihood over all cells.

    Assumes cells are independent given the forecast. That is false — real
    seismicity is spatially correlated through aftershock sequences — and it is
    the standard assumption anyway, because the alternative requires specifying
    the correlation structure, which is most of the forecasting problem.
    """
    rates, counts, _ = _prepare(forecast, observed)
    return float(np.sum(-rates + counts * np.log(rates) - gammaln(counts + 1.0)))


def n_test(forecast: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    """Number test. Returns (delta1, delta2).

    Under the forecast, the total count is Poisson with mean `sum(forecast)`.

    - `delta1` = P(N >= N_obs). Small means the model forecast far too few.
    - `delta2` = P(N <= N_obs). Small means it forecast far too many.

    At the conventional two-sided 5% level, reject if either is below 0.025.
    Both being comfortably above it is the pass condition.
    """
    rates, counts, _ = _prepare(forecast, observed)
    n_fore = float(np.sum(rates))
    n_obs = int(np.sum(counts))

    delta1 = float(poisson.sf(n_obs - 1, mu=n_fore))
    delta2 = float(poisson.cdf(n_obs, mu=n_fore))
    return delta1, delta2


def s_test(
    forecast: np.ndarray,
    observed: np.ndarray,
    *,
    n_simulations: int = N_SIMULATIONS,
    seed: int | None = 0,
) -> float:
    """Spatial test. Returns gamma.

    The total count is removed as a variable by rescaling the forecast so it sums
    to the observed number of events; what remains is purely the spatial
    distribution. `N_obs` events are then dropped onto cells multinomially
    according to the normalised forecast, many times, and the observed spatial
    log-likelihood is compared against that null.

    `gamma` is the fraction of simulations at or below the observed likelihood.
    Small gamma means the real events landed somewhere the model considered
    unlikely — a spatial failure. Reject below 0.05 one-sided.

    Seeded by default so a leaderboard entry is reproducible; pass `seed=None`
    for an independent draw.
    """
    rates, counts, _ = _prepare(forecast, observed)
    n_obs = int(np.sum(counts))
    total = float(np.sum(rates))

    if n_obs == 0 or total <= 0:
        # No events, or a degenerate forecast: the spatial distribution is not
        # testable. NaN rather than a pass, so it cannot be read as one.
        return float("nan")

    scaled = rates * (n_obs / total)
    scaled = np.maximum(scaled, RATE_FLOOR)

    observed_ll = float(np.sum(-scaled + counts * np.log(scaled) - gammaln(counts + 1.0)))

    rng = np.random.default_rng(seed)
    probabilities = scaled / np.sum(scaled)
    simulated = rng.multinomial(n_obs, probabilities, size=n_simulations).astype(float)

    log_scaled = np.log(scaled)
    sim_ll = np.sum(-scaled + simulated * log_scaled - gammaln(simulated + 1.0), axis=1)

    return float(np.mean(sim_ll <= observed_ll))


def brier_score(forecast: np.ndarray, observed: np.ndarray) -> float:
    """Mean Brier score for "at least one event in this cell".

    The Poisson probability of a non-empty cell is `1 - exp(-lambda)`. Lower is
    better; 0 is perfect. Most cells are empty in any short window, so the
    absolute value is always small — it is only meaningful compared against
    another model on the same windows.
    """
    rates, counts, _ = _prepare(forecast, observed)
    probability = 1.0 - np.exp(-rates)
    outcome = (counts > 0).astype(float)
    return float(np.mean((probability - outcome) ** 2))


def score(
    forecast: np.ndarray,
    observed: np.ndarray,
    *,
    n_simulations: int = N_SIMULATIONS,
    seed: int | None = 0,
) -> Scores:
    """Run every test and return them together."""
    rates, counts, n_floored = _prepare(forecast, observed)
    delta1, delta2 = n_test(rates, counts)
    return Scores(
        log_likelihood=log_likelihood(rates, counts),
        n_test_delta1=delta1,
        n_test_delta2=delta2,
        s_test_gamma=s_test(rates, counts, n_simulations=n_simulations, seed=seed),
        brier=brier_score(rates, counts),
        n_forecast=float(np.sum(rates)),
        n_observed=int(np.sum(counts)),
        n_floored_cells=n_floored,
    )
