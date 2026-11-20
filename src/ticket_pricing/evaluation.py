"""
Out-of-sample evaluation of the tuned policies on a fresh block of >=500
common-random-number scenarios (EVAL_SEEDS), disjoint from the
TUNING_SEEDS used to pick each policy's theta. Reports mean, standard
deviation, 5th/95th percentile spread and a 95% CI on the mean (via the
normal approximation to the sample mean, N large enough at 500+ draws).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ticket_pricing.calibration import Calibration
from ticket_pricing.simulator import run_batch

EVAL_SEEDS = list(range(200_000, 200_600))   # 600 scenarios >= the 500 required, disjoint from tuning


@dataclass
class EvalResult:
    name: str
    mean: float
    std: float
    p05: float
    p95: float
    ci95_low: float
    ci95_high: float
    n: int
    revenues: np.ndarray


def evaluate(policy, name: str, calib: Calibration, seeds: list[int] | None = None) -> EvalResult:
    seeds = seeds or EVAL_SEEDS
    rev = run_batch(policy, calib, seeds)
    mean = float(rev.mean())
    std = float(rev.std(ddof=1))
    se = std / np.sqrt(len(seeds))
    return EvalResult(
        name=name,
        mean=mean,
        std=std,
        p05=float(np.percentile(rev, 5)),
        p95=float(np.percentile(rev, 95)),
        ci95_low=mean - 1.96 * se,
        ci95_high=mean + 1.96 * se,
        n=len(seeds),
        revenues=rev,
    )


def summarize(results: list[EvalResult]) -> str:
    lines = [f"{'policy':10s} {'mean':>10s} {'std':>9s} {'p05':>9s} {'p95':>9s} {'95% CI':>22s} {'n':>5s}"]
    for r in results:
        ci = f"[{r.ci95_low:8.1f}, {r.ci95_high:8.1f}]"
        lines.append(f"{r.name:10s} {r.mean:10.2f} {r.std:9.2f} {r.p05:9.2f} {r.p95:9.2f} {ci:>22s} {r.n:5d}")
    return "\n".join(lines)
