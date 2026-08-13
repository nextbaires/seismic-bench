# Repository-specific drafting instructions

Read this before implementing a proposal here. It takes precedence over the
generic expert-loop prompt.

## The one rule that outranks everything

**Never let a model see data from its own forecast window.** Every model receives
`Window.training_events(catalog)` and nothing else. Do not add a parameter, a
keyword argument, or a "just for calibration" path that widens it. Do not fit
anything — a completeness magnitude, a smoothing bandwidth, an ETAS parameter —
on the full catalog and then use it inside a window.

If a proposal seems to require full-catalog access, it almost certainly has a
per-window formulation instead. Implement that one. If it genuinely does not,
stop and say so on the issue rather than writing the leaky version.

`tests/test_split.py` enforces this. Any change that makes those tests pass by
weakening them is wrong.

## Scientific claims need sources

Any assertion about the Earth — as opposed to an assertion about this code —
carries a DOI or URL in the comment or docstring where it appears. "Aftershock
productivity scales as 10^(alpha*M)" needs a citation. "This loop iterates over
grid cells" does not.

If the proposer gave firsthand field experience rather than a paper, attribute it
to them explicitly: `# Per @username (#42): regional networks in the Andes
retune this by mechanism.` That is legitimate evidence and should be visible as
what it is.

## Models

A new model implements the `Forecaster` protocol in
`seismic_bench/baselines/base.py` and nothing more. Register it in `REGISTRY` in
`seismic_bench/bench/run.py`.

Return expected *counts per cell for the window*, not rates per day and not
probabilities. Length `grid.n_cells`, flattened lat-major, non-negative, finite.

## Tests

A change to the science needs a test on the behaviour, not on the plumbing. A
test asserting a function returns an array of the right shape does not
demonstrate that a new decay law is correct; a test asserting the forecast is
higher immediately after a mainshock than three months later does.

Do not assert exact leaderboard scores in tests. That freezes the baselines
against improvement, which is the opposite of the point.

## What not to touch

`seismic_bench/bench/split.py`, `metrics.py`, and `grid.py` define the protocol
every published result depends on. Changing them silently invalidates the
leaderboard. If a proposal genuinely requires a protocol change, say so on the
issue and let a maintainer decide — do not implement it.
