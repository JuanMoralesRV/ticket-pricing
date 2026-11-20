# Project B -- Dynamic ticket pricing

A Powell five-element sequential-decision model for pricing seats on a single
flight, with three policies (a fixed-price baseline, a linear Policy Function
Approximation, and a Direct Lookahead Approximation), calibrated from real
fare data and evaluated with common random numbers.

See `reports/report.md` for the full write-up: model, calibration, results,
and the recommendation. This README is just the "how to run it" half.

## Project layout

```
src/ticket_pricing/
  calibration.py     fit the model's parameters from data/itineraries_sample.csv
  model.py            the Powell model: state, decision, exogenous info, transition, objective
  simulator.py         gym-style step() environment + CRN episode/batch runner
  tuning.py            policy search (max_theta F_bar(theta)) for baseline & PFA
  evaluation.py        out-of-sample evaluation on >=500 CRN scenarios, CI/spread
  policies/
    baseline.py         fixed-price policy
    pfa.py               linear Policy Function Approximation
    dla.py               Direct Lookahead Approximation (deterministic MPC-style)
dashboard/app.py      Streamlit dashboard (interactive version of the report's figures)
scripts/run_experiment.py   reproduces every number and figure in the report
notebooks/            exploratory data analysis (optional, for your own reference)
tests/                 pytest sanity checks on the model and policies
data/                  the fare sample + calibration.json (generated)
reports/               results.json + the .png figures + report.md (generated / written)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

## Reproduce everything

```bash
# 1. (re)fit the calibration from the fare sample (already included in data/)
python src/ticket_pricing/calibration.py

# 2. tune both policies, evaluate all three, write reports/results.json + figures
python scripts/run_experiment.py

# 3. run the test suite
pytest tests/ -q

# 4. launch the interactive dashboard
streamlit run dashboard/app.py
```

## Data note

`data/itineraries_sample.csv` is a ~1M-row sample (not the full 31 GB file)
of Kaggle's `dilwong/flightprices` dataset, pulled directly from the
compressed file's byte stream to avoid downloading the whole thing -- see
the docstring at the top of `calibration.py` for exactly what came from the
data versus what was assumed, and why.

## The three policies, in one line each

- **Baseline (fixed price)**: one constant price for the whole horizon, tuned
  by a 1-D grid search. The yardstick every other policy has to beat.
- **PFA (linear)**: `price = theta_a + theta_b*scarcity_term + theta_c*time_term`,
  evaluated directly from the state, no optimization at decision time. Tuned
  by random search + local refinement (max_theta F_bar(theta), common random
  numbers).
- **DLA**: at every decision, solves a small deterministic (expected-value)
  model of the rest of the horizon over a price grid, posts only that day's
  price, and re-solves from scratch next period (model-predictive-control
  pattern).

## Headline result (see report for the full discussion)

On the real-data calibration, the tuned baseline and the tuned PFA price
schedule end up identical (PFA finds theta_b = theta_c = 0 is optimal) and
DLA is statistically indistinguishable from both. Under a higher-demand
sensitivity scenario, DLA pulls ahead because it can track the calibrated,
non-monotonic WTP-vs-time curve, which the linear PFA cannot represent. The
report explains why and gives the conditions under which the recommendation
would change.
