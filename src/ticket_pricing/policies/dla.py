"""Direct Lookahead Approximation: at every decision point, solve an
approximate (deterministic, expected-value) model of the remaining booking
horizon, then implement only the first decision and re-solve from scratch
next period -- the model-predictive-control pattern from 007_Policy_Classes.

The approximation: replace the stochastic Poisson-arrivals-plus-WTP-draws
process with its expectation. For a candidate constant price p held over
the remaining horizon, the expected seats sold on a future day with
remaining-capacity fraction f is

    lambda(f) * P(WTP >= hurdle(p))

where lambda(f) is the (deterministic) scarcity-adjusted arrival rate and
P(WTP >= hurdle(p)) is the lognormal survival function evaluated at the
loss-aversion-adjusted hurdle. We roll this forward day by day with a
continuous (non-integer) capacity, pick the price that maximizes total
expected revenue over the lookahead window, post only that price today, and
discard the rest of the plan -- next period we lookahead again from the
real, realized state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc

from ticket_pricing.calibration import Calibration, wtp_mu_at
from ticket_pricing.model import State, purchase_hurdle

_SQRT2 = 2.0 ** 0.5


def _lognormal_sf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """P(X >= x) for X ~ Lognormal(mu, sigma), vectorized, without the
    overhead of building a scipy.stats frozen-distribution object (this
    runs once per lookahead day, for every candidate price in the grid)."""
    x = np.maximum(x, 1e-12)
    return 0.5 * erfc((np.log(x) - mu) / (sigma * _SQRT2))


def _deterministic_lookahead_values(
    prices: np.ndarray, state: State, calib: Calibration, lookahead_days: int
) -> np.ndarray:
    """Vectorized over the whole price grid at once: for every candidate
    price, roll the deterministic (expected-value) model forward day by day.
    The day loop stays sequential (capacity depletion is path-dependent),
    but all grid points for a given day are evaluated in one numpy call.
    Uses the SAME calibrated time-varying WTP location (wtp_mu_at) as the
    real simulator -- an inconsistent lookahead model would make this an
    unfair test of whether looking ahead is worth it."""
    prices = np.asarray(prices, dtype=float)
    c = np.full_like(prices, float(state.capacity))
    r = np.full_like(prices, state.ref_price)
    revenue = np.zeros_like(prices)
    remaining = min(lookahead_days, calib.horizon_days - state.t)
    for day in range(remaining):
        alive = c > 1e-9
        if not alive.any():
            break
        frac = c / calib.capacity
        lam = np.where(frac < calib.scarcity_threshold,
                        calib.arrival_rate_base * calib.kappa_scarcity_multiplier,
                        calib.arrival_rate_base)
        hurdle = prices + calib.eta_loss_aversion * np.maximum(prices - r, 0.0) \
                        - calib.gamma_gain * np.maximum(r - prices, 0.0)
        lead_days_remaining = calib.horizon_days - (state.t + day)
        mu_t = wtp_mu_at(calib, lead_days_remaining)
        p_buy = _lognormal_sf(hurdle, mu_t, calib.wtp_sigma)
        expected_sales = np.minimum(lam * p_buy, c) * alive
        revenue += prices * expected_sales
        c -= expected_sales
        r = calib.theta_ref * r + (1 - calib.theta_ref) * prices
    return revenue


@dataclass
class DirectLookaheadPolicy:
    calib: Calibration
    lookahead_days: int = 60          # how many future days the deterministic model simulates
    n_grid: int = 25                   # price-grid resolution solved at every decision
    price_grid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.price_grid = np.linspace(self.calib.price_floor, self.calib.price_ceiling, self.n_grid)

    def price(self, state: State) -> float:
        values = _deterministic_lookahead_values(self.price_grid, state, self.calib, self.lookahead_days)
        return float(self.price_grid[int(np.argmax(values))])
