# seismic-bench

**An open, reproducible benchmark for earthquake forecasting — and a way for
seismologists to contribute to it without writing code.**

Fetch a catalog, score a model against fixed baselines on a protocol designed to
make lookahead impossible, and publish the result. Everything runs offline from a
committed sample, so a clone reproduces the leaderboard on a laptop with no API
keys and no downloads.

---

## What this is honest about

Earthquake forecasting attracts more overclaiming than almost any field in
geoscience, so it is worth being blunt up front.

**Deterministic earthquake prediction — this fault, this day, this magnitude — is
not a solved problem, and this repository does not attempt it.** Decades of
searching for reliable short-term precursors have not produced one that survives
prospective testing. If you came here looking for that, this is not it, and you
should be suspicious of anything that says otherwise.

**What does work is probabilistic forecasting**, and it works well enough to be
operationally useful. Aftershock sequences follow reproducible statistics.
Seismicity clusters in space in ways that persist over decades. A good model can
tell you that the rate in a region over the next month is fifteen times its
long-run average, and be right often enough to matter for emergency planning,
insurance, and building-code policy. That is a real, measurable, improvable
thing, and it is what gets scored here.

**The baselines are hard to beat and that is the point.** Smoothed historical
seismicity — "earthquakes happen where earthquakes have happened" — is a strong
forecaster. A great deal of published skill is that model wearing a hat. If your
approach does not beat it out of sample on these windows, it has not shown
anything yet.

**The ETAS baseline here is deliberately not fit to the data.** It runs on fixed
literature parameters, which is worse than a proper per-window maximum-likelihood
fit and is documented as such in
[`seismic_bench/baselines/etas.py`](seismic_bench/baselines/etas.py). Fixing that
is probably the single most valuable contribution this repository could receive.

## The protocol

A model receives the events before a window opens and returns an expected count
per grid cell for that window. That is the entire interface.

| | |
|---|---|
| **Region** | California, `-125..-113` E, `31..43` N (roughly the CSEP RELM region) |
| **Grid** | 0.2 degree cells, 3,600 of them |
| **Window** | 30 days, consecutive, non-overlapping |
| **Training** | Expanding, everything before the window opens, minimum 3 years |
| **Magnitude** | M ≥ 3.0 by default, above California's completeness threshold |
| **Scoring** | Poisson log-likelihood, CSEP N-test and S-test, Brier score |

**Lookahead is structurally prevented, not just avoided.** A model is handed a
`Window` object whose `training_events` method is constructed to end strictly
before the forecast window opens. There is no argument that widens it and no
handle on the full catalog. Boundaries are half-open on both ends, so an event at
the exact instant a window opens belongs to the test side — the side that does
not help. [`tests/test_split.py`](tests/test_split.py) asserts all of this,
including on adversarially-constructed windows.

The metrics are the CSEP consistency tests rather than something cleaner of our
own, because using the tests the field already argues about is worth more than
inventing better ones nobody trusts. They answer different questions and a model
can pass one while failing another: the N-test asks whether you forecast the
right *number* of events, the S-test whether you got the *spatial pattern* right
once the total is controlled for. A regional-rate model passes the first and
fails the second; a smoothed-seismicity model often does the reverse.

## Quick start

```bash
git clone https://github.com/nextbaires/seismic-bench
cd seismic-bench
uv venv && uv pip install -e ".[dev]"

# Runs on the committed sample — no network needed.
python -m seismic_bench.bench.run --model poisson
python -m seismic_bench.bench.run --model etas
pytest
```

For a real run, fetch a full catalog first:

```bash
python -m seismic_bench.ingest.cli --start 2000-01-01 --end 2024-01-01
python -m seismic_bench.bench.run --model etas
```

## Submitting a model

Implement one method:

```python
class MyModel:
    name = "my-model"

    def forecast(self, training_events, grid, window):
        """Expected event counts per cell. Shape (grid.n_cells,), non-negative."""
        ...

    def describe(self):
        """Parameters, for the leaderboard record."""
        return {"model": "MyModel", ...}
```

Then either add it to `REGISTRY` in
[`seismic_bench/bench/run.py`](seismic_bench/bench/run.py) or run it directly:

```bash
python -m seismic_bench.bench.run --model mypackage.models:MyModel
```

Open a pull request with the model and its leaderboard JSON.

## Contributing without writing code

**If you are a seismologist and not a programmer, you can still contribute — and
your contribution is the kind this project most needs.**

Open an issue using the *Domain proposal* template. Describe what you know: what
the model gets wrong, what the literature says, what your own field experience
tells you, and how we would know whether you are right. You do not need to read
the code or know Python.

A maintainer reads it. If it is in scope, an assistant implements it against this
repository, writes a test, cites your sources, and opens a pull request with your
proposal quoted verbatim so you can check it says what you meant. You are credited
as the proposer. The domain judgement stays yours; if the implementation is wrong,
that is our bug to fix, not yours.

This runs on [`nextbaires/expert-loop`](https://github.com/nextbaires/expert-loop).

Things that would genuinely help, if you are looking for somewhere to start:

- **Per-window maximum-likelihood ETAS fitting.** The single biggest gap.
- **Adaptive-bandwidth smoothing** for the Poisson baseline, following
  Helmstetter, Kagan & Jackson (2007), which is known to beat the fixed-bandwidth
  version used here.
- **Time-varying magnitude of completeness.** Mc rises sharply during aftershock
  sequences as small events are masked; treating it as constant biases exactly
  the windows that matter most.
- **Declustering**, done correctly. Any leak-free way to separate background from
  triggered seismicity within the rolling protocol.
- **Regions beyond California.** Italy (INGV) and Japan (JMA) have catalogs good
  enough to benchmark on, and different enough tectonics to be a real test.
- **Telling us the baselines are wrong.** If the fixed ETAS parameters are
  inappropriate for this region, that is a finding, and it is one an expert can
  report in two paragraphs.

## Data

Catalogs come from the USGS ComCat / ANSS service. Code is MIT; derived data is
CC BY 4.0; source data carries its own terms. See [LICENSE-DATA](LICENSE-DATA) —
if you redistribute a catalog built with this tool, carry the attribution
through. The people operating the seismic networks are why any of this exists.

The committed sample under `data/sample/` is real USGS data, not simulated, kept
small enough for git so that tests and CI run offline.

## Leaderboard

Current standings on the committed sample — California, M ≥ 3.0, 36 consecutive
30-day windows from June 2017 to December 2019. Reproduce with `pytest &&
python -m seismic_bench.bench.run --model poisson --model etas`.

| Rank | Model | Total log-likelihood | N-test pass | S-test pass |
|---|---|---:|---:|---:|
| 1 | `poisson` | −13,256.56 | 69% | 83% |
| 2 | `etas` | −14,247.98 | 39% | 61% |

**The time-dependent model loses to the time-independent one, and that is the
most useful thing on this page.** It is what the unfit-parameters warning above
predicts, and it is a concrete, well-specified problem for somebody to fix.

The aggregate hides something important, though. In the window immediately after
the July 2019 Ridgecrest M7.1, the ranking inverts hard:

| Window opening 2019-07-19 | Forecast events | Observed | Log-likelihood |
|---|---:|---:|---:|
| `poisson` | 41.7 | 73 | −266.26 |
| `etas` | 105.8 | 73 | **−127.20** |

ETAS is doing exactly what a triggering model should — it saw the mainshock and
raised its forecast an order of magnitude while smoothed seismicity carried on
predicting the long-run average. It then loses that advantage back over quiet
months by decaying too fast, ending up worse overall.

So the honest summary is: **ETAS is better when it matters and worse on average,
and fixing that is a parameter-estimation problem, not a modelling one.** If you
know how to fit ETAS properly per window, you would be fixing the most valuable
open problem in this repository, and you do not have to write the code yourself.

Full records — every window, every model's parameters, the environment — are in
[`leaderboard/results.json`](leaderboard/results.json), rebuilt by CI on every
merge to `main`. CI fails if the committed results drift from what the code
produces, so a number nobody can reproduce cannot appear here.

## License

MIT for code, CC BY 4.0 for derived data. See [LICENSE](LICENSE) and
[LICENSE-DATA](LICENSE-DATA).
