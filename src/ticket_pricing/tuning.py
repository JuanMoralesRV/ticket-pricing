"""
Policy search: max_theta F_bar(theta) = (1/N) * sum_n C(S^n, X^pi(S^n | theta))

Both policy classes get tuned on their OWN block of seeds (TUNING_SEEDS),
kept completely separate from the EVAL_SEEDS used later in evaluation.py.
Tuning and reporting on the same scenarios is exactly the pitfall
008_PFA_and_CFA calls out ("tuning and reporting on the same scenarios:
test on fresh seeds") -- using two disjoint seed blocks is how we avoid it.
Every theta candidate within one tuning run is scored on the identical seed
list (common random numbers), so differences in F_bar(theta) are driven by
theta, not by which customers happened to show up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ticket_pricing.calibration import Calibration
from ticket_pricing.simulator import run_batch
from ticket_pricing.policies.baseline import FixedPricePolicy
from ticket_pricing.policies.pfa import LinearPFA

TUNING_SEEDS = list(range(100_000, 100_100))   # 100 scenarios, disjoint block from eval


@dataclass
class TuningResult:
    theta: dict
    mean_revenue: float
    std_revenue: float
    n_candidates_tried: int


def tune_baseline(calib: Calibration, n_grid: int = 25, seeds: list[int] | None = None) -> TuningResult:
    seeds = seeds or TUNING_SEEDS
    grid = np.linspace(calib.price_floor, calib.price_ceiling, n_grid)
    best = None
    for p in grid:
        rev = run_batch(FixedPricePolicy(price_level=float(p)), calib, seeds)
        if best is None or rev.mean() > best[1]:
            best = (float(p), rev.mean(), rev.std())
    return TuningResult(theta={"price_level": best[0]}, mean_revenue=best[1], std_revenue=best[2],
                         n_candidates_tried=n_grid)


def _score(theta_a, theta_b, theta_c, calib, seeds):
    pol = LinearPFA(theta_a=theta_a, theta_b=theta_b, theta_c=theta_c, calib=calib)
    rev = run_batch(pol, calib, seeds)
    return float(rev.mean()), float(rev.std())


def tune_pfa(
    calib: Calibration,
    n_candidates: int = 250,
    seeds: list[int] | None = None,
    rng_seed: int = 7,
    seed_price: float | None = None,
    n_refine_passes: int = 2,
) -> TuningResult:
    """Random search over (theta_a, theta_b, theta_c), seeded with the
    degenerate theta_b=theta_c=0 point (a pure constant price recovers the
    FixedPricePolicy exactly, since this PFA's two markup terms are each
    >= 0 by construction) so the search is never worse than the baseline's
    own best price -- then a short coordinate-wise local refinement, since
    a 3-parameter grid fine enough to pinpoint the optimum would need
    n_grid**3 evaluations (the "grid search cost grows exponentially"
    warning from 008_PFA_and_CFA). Candidates and refinement steps are all
    scored on the same CRN seeds, so every comparison is apples to apples."""
    seeds = seeds or TUNING_SEEDS
    rng = np.random.default_rng(rng_seed)
    span = calib.price_ceiling - calib.price_floor

    candidates = []
    if seed_price is not None:
        candidates.append((seed_price, 0.0, 0.0))
    candidates.append((calib.price_floor, span, span))
    candidates.append((calib.wtp_median, span / 2, span / 2))
    for _ in range(n_candidates):
        candidates.append((
            rng.uniform(calib.price_floor, calib.price_ceiling),
            rng.uniform(0.0, span),
            rng.uniform(0.0, span),
        ))

    best = None
    for ta, tb, tc in candidates:
        mean_rev, std_rev = _score(ta, tb, tc, calib, seeds)
        if best is None or mean_rev > best[1]:
            best = ((ta, tb, tc), mean_rev, std_rev)
    tried = len(candidates)

    # local coordinate-wise refinement around the best candidate found
    (ta, tb, tc), mean_rev, std_rev = best
    step = span / 8
    for _ in range(n_refine_passes):
        improved = False
        for idx in range(3):
            cur = [ta, tb, tc]
            lo_bound = calib.price_floor if idx == 0 else 0.0
            hi_bound = calib.price_ceiling if idx == 0 else span
            for delta in (-step, step):
                trial = cur.copy()
                trial[idx] = min(max(trial[idx] + delta, lo_bound), hi_bound)
                m, s = _score(*trial, calib, seeds)
                tried += 1
                if m > mean_rev:
                    ta, tb, tc = trial
                    mean_rev, std_rev = m, s
                    improved = True
        step /= 2
        if not improved:
            break

    return TuningResult(
        theta={"theta_a": ta, "theta_b": tb, "theta_c": tc},
        mean_revenue=mean_rev,
        std_revenue=std_rev,
        n_candidates_tried=tried,
    )
