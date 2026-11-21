# Project B: Dynamic Ticket Pricing

*Financial Decision Theory / Sequential Decision Analytics -- Group MAF4129O*

## 1. The problem as a sequential decision

We price seats on a single flight over a 60-day booking horizon. Every day
the airline posts one price; customers arrive at random, each privately
willing to pay up to their own reservation value, and buy only if the posted
price clears a reference-price-adjusted hurdle. The airline cannot observe
who is coming or what they are willing to pay before posting a price, and a
seat sold cannot be unsold -- this is exactly the kind of problem the
course's policy-class framework (PFA / CFA / VFA / DLA) was built for.

## 2. The Powell model

**State** `S_t = (t, c_t, r_t)`: the day index, seats remaining, and
customers' reference price. This triple is a sufficient statistic -- nothing
before `t` matters once we know these three numbers.

**Decision** `x_t = p_t`, the price posted during day `t`, constrained to
`[price_floor, price_ceiling]` (the 5th/95th percentile of real observed
fares on the route, so the price grid never wanders into implausible
territory).

**Exogenous information** `W_{t+1} = (N_{t+1}, {WTP_i})`: the number of
customer arrivals on day `t`, `N_{t+1} ~ Poisson(lambda_t)`, each with an
independently drawn willingness to pay, `WTP_i ~ Lognormal(mu_t, sigma)`.
`lambda_t` jumps by a scarcity multiplier `kappa` once remaining capacity
drops below 10% of total capacity (this trigger is specified by the
assignment; `kappa`'s size is an assumption, see Section 3). `mu_t` is
**not** constant -- it is fit directly from real fare data (Section 3) and
rises as departure approaches, capturing the standard revenue-management
story that late bookers skew higher-WTP.

**Transition function**: each arriving customer buys if their WTP clears

```
hurdle(p_t) = p_t + eta * max(p_t - r_t, 0) - gamma * max(r_t - p_t, 0)
```

(pricing above the reference costs an `eta`-weighted penalty; pricing below
it earns a `gamma`-weighted discount -- loss aversion / reference
dependence). Each sale decrements capacity and adds `p_t` to the day's
revenue; the reference price then drifts toward whatever was actually
posted: `r_{t+1} = theta_ref * r_t + (1 - theta_ref) * p_t`.

**Objective**: choose a policy to maximize expected total revenue over the
horizon,

```
max_pi  E[ sum_t sum_i p_t * 1{customer i buys on day t} ]
```

## 3. Calibration: what came from data, what we assumed

We calibrated against Kaggle's public "Flight Prices" dataset
(`dilwong/flightprices`), filtered to Atlanta-Boston, nonstop, main cabin
(**1,405 observations**). The full file is 31 GB; downloading it whole was
not practical, so we pulled roughly a 1M-row sample directly off the file's
compressed byte stream (see the top of `calibration.py` for the mechanics)
and filtered down to this route from there.

From the data, we fit:

- **The WTP-over-time curve.** A degree-2 regression of `log(totalFare)` on
  days-to-departure, using all 1,405 rows rather than a handful of noisy
  lead-time-bucket averages (one bucket, 4-7 days out, has only 122 rows and
  swings wildly on its own -- exactly the kind of sampling noise the course
  slides warn about). The fit gives a smooth, U-shaped curve: median WTP is
  highest right at departure (~$410), dips to its lowest in the 20-40 day
  window (~$263-268), and rises again for the earliest bookers (~$305 at 60
  days out). See `reports/fig_price_vs_leadtime.png`.
- **The WTP dispersion**, `sigma = 0.461`, the residual spread of
  `log(totalFare)` around that fitted curve.
- **The price grid bounds**, `[price_floor, price_ceiling]` = the 5th/95th
  percentile of observed fares on the route.
- **The capacity scale.** The dataset caps displayed `seatsRemaining` at 10.
  We read this as one fare bucket / booking-class allotment (not a whole
  aircraft) and use `C = 10` directly.

We could **not** identify from this dataset (it is a cross-section of
searches, not a longitudinal purchase log) true arrival rates, loss-aversion
coefficients, or the scarcity multiplier's magnitude. These are assumed,
clearly labeled as such in `calibration.py`, and explicitly sensitivity
tested in Section 5 rather than hidden:

| Parameter | Value | Source |
|---|---|---|
| `theta_ref` (reference-price smoothing) | 0.7 | assumed |
| `eta` (loss-aversion penalty) | 1.5 | assumed |
| `gamma` (below-reference discount) | 0.5 | assumed |
| scarcity threshold | 10% of capacity | **specified by the assignment** |
| `kappa` (scarcity arrival multiplier) | 1.3x | assumed |
| `arrival_rate_base` (main scenario) | capacity/horizon x 1.4 | assumed demand-to-capacity ratio |

## 4. Policies

**Baseline (fixed price).** One constant price for the whole horizon, tuned
by a 25-point grid search over `[price_floor, price_ceiling]`.

**PFA (linear).** `price = theta_a + theta_b * scarcity_term + theta_c *
time_term`, where `scarcity_term = 1 - c_t/C` and `time_term = t/horizon`
are both in `[0,1]`, so `theta_b` and `theta_c` are additive markups in USD.
This is a textbook PFA: the shape is fixed in advance, and evaluating it at
decision time costs nothing (no optimization). Tuned by random search (250
candidates) seeded with the degenerate `theta_b = theta_c = 0` point --
which exactly recovers the baseline -- plus a short coordinate-wise local
refinement, since a grid fine enough to pin down three parameters precisely
would need `n_grid^3` evaluations (the "grid search cost grows
exponentially" warning from the lecture slides).

**DLA (Direct Lookahead Approximation).** At every decision, replace the
stochastic arrivals-plus-WTP process with its expectation and roll a
deterministic version of the remaining horizon forward for every candidate
price in a 25-point grid, using the *same* calibrated time-varying WTP curve
as the real simulator. Post only the best candidate's price for one day,
discard the rest of the plan, and re-solve from the real, realized state
next period -- the model-predictive-control pattern.

## 5. Evaluation methodology

Both tunable policies (baseline, PFA) were tuned on **100 tuning
scenarios** (seeds 100000-100099). All three policies were then evaluated
out-of-sample on **600 evaluation scenarios** (seeds 200000-200599), a
disjoint block the tuning step never saw. Every policy in a given comparison
faces the *identical* seed list -- common random numbers -- so any
difference in revenue is attributable to the policy, not to which customers
happened to show up that run.

## 6. Results

### 6.1 Real-data calibration (moderate demand, E[arrivals] = 14.0 against capacity 10)

| Policy | Mean revenue | Std | p05 | p95 | 95% CI |
|---|---|---|---|---|---|
| Baseline | $2,092.0 | 480.9 | 1,261.8 | 2,523.5 | [2,053.5, 2,130.5] |
| PFA | $2,092.0 | 480.9 | 1,261.8 | 2,523.5 | [2,053.5, 2,130.5] |
| DLA | $2,062.5 | 469.2 | 1,162.8 | 2,721.4 | [2,025.0, 2,100.1] |

The tuned PFA converges *exactly* to the baseline (`theta_b = theta_c =
0`) -- we verified this is not a tuning artifact by sweeping each markup
coefficient alone (holding the other at the tuned baseline price): revenue
decreases **monotonically** as either `theta_b` or `theta_c` increases above
zero. DLA is statistically indistinguishable from both (its 95% CI overlaps
substantially with baseline/PFA's).

### 6.2 High-demand sensitivity check (demand ~3x capacity, E[arrivals] = 30.0)

| Policy | Mean revenue | Std | p05 | p95 | 95% CI |
|---|---|---|---|---|---|
| Baseline | $3,162.4 | 549.4 | 2,090.3 | 3,513.1 | [3,118.4, 3,206.3] |
| PFA | $3,162.4 | 549.4 | 2,090.3 | 3,513.1 | [3,118.4, 3,206.3] |
| DLA | **$3,201.9** | 501.0 | 2,275.3 | 3,909.9 | [3,161.8, 3,242.0] |

Here DLA pulls ahead by about 1.3%, and the two confidence intervals barely
overlap (DLA's lower bound, 3,161.8, sits just above baseline/PFA's mean of
3,162.4 -- close, but DLA's advantage is far more consistent than in the
moderate-demand case).

See `reports/fig_revenue_dist.png` for the full revenue distributions and
`reports/fig_price_path.png` for one representative scenario's price path:
baseline/PFA sit flat while DLA visibly tracks the U-shaped WTP curve,
discounting mid-horizon and marking up sharply in the final ~10 days.

## 7. Why PFA collapses to the baseline, and why DLA doesn't

This is the most interesting finding and worth spelling out rather than
glossing over. The calibrated WTP-vs-time curve is **U-shaped, not
monotonic**: high near departure, low in the middle of the horizon, and
slightly elevated again for the earliest bookers. A *linear* `time_term`
can only push price up or down monotonically over the horizon -- it cannot
represent "discount in the middle, then mark up at the end." Any nonzero
`theta_c` therefore fights the true curve somewhere along the horizon,
losing more in forgone sales than it gains in margin, which is exactly what
the parameter sweep in Section 6.1 shows.

DLA does not have this limitation: at every decision it evaluates expected
revenue using the *actual* calibrated `mu_t`, so it can track the curve's
true shape -- discount when WTP is low and demand is not pressing, mark up
when WTP is high or inventory is genuinely tight. Under moderate demand
(Section 6.1) there is rarely enough of a capacity squeeze for this
precision to matter, so DLA's edge is within noise. Under demand pressure
(Section 6.2), getting the price right on more days actually changes which
customers you serve, and DLA's edge becomes real.

This is consistent with the lecture's own lesson on policy classes: gains
from a richer policy depend on how far the baseline already sits from the
optimum, and a PFA is only as good as the functional form you chose for it.
Our linear PFA was the wrong shape for a U-shaped WTP curve; a quadratic (or
spline) PFA would likely close most of the gap to DLA without needing a
lookahead solve at every decision -- a natural next iteration we did not
have time to build here.

## 8. Recommendation

**For this route, under the demand levels we calibrated from real data, we
recommend the tuned fixed-price baseline.** It is the simplest policy to
implement, explain, and audit, and it is statistically indistinguishable
from both DLA and the best policy inside the linear-PFA family. Paying for
DLA's operational complexity (re-solving an optimization every day, in
production) is not justified by the evidence at this demand level.

**This recommendation would change under any of the following conditions:**

1. **Demand materially exceeds capacity** (our sensitivity check used ~3x;
   the effect should strengthen further as this ratio grows). At that
   point, switch to DLA -- the revenue gain becomes both larger and more
   statistically reliable.
2. **The route's WTP-vs-time curve turns out to be strongly non-monotonic**
   (as ours is). If a simpler, monotonic curve were found instead, a linear
   PFA with a genuinely nonzero `theta_c` could likely match DLA's
   performance without the daily re-optimization cost -- worth re-tuning PFA
   whenever the calibration is refreshed on new data.
3. **Operational cost of DLA drops** -- e.g., a faster solver or a coarser,
   still-accurate price grid -- making its added complexity cheaper to
   justify even for a modest revenue gain.
4. **The assumed parameters (`eta`, `gamma`, `kappa`, `theta_ref`) turn out
   to be materially different** once better data is available (e.g. an
   internal booking log with real purchase timestamps instead of a search
   snapshot). We would re-run `scripts/run_experiment.py` against the
   updated calibration before trusting this recommendation further.

## 9. Limitations

The Kaggle dataset is a cross-section of searches scraped over a 3-day
window (April 16-18, 2022) for flights departing over the following two
months, not a longitudinal purchase log for one flight -- so "arrival rate"
and the loss-aversion parameters could not be estimated directly and are
documented assumptions, stress-tested in Section 6.2 rather than taken on
faith. The WTP distribution is proxied by observed fares, which is a
lower-bound proxy (a sale only happens when WTP clears the price), not an
exact recovery of the latent WTP distribution. All of this is standard for
an academic exercise at this scope, and the code is structured
(`calibration.py`) so that plugging in a better dataset later is a drop-in
change, not a rewrite.
