"""
Powell's five-element sequential-decision model for dynamic ticket pricing.

We are selling seats for a single flight departing in `horizon_days` days.
Every day the airline posts one price; zero, one or several customers may
arrive that day, each privately drawing a willingness to pay (WTP) and
buying only if the posted price clears a reference-price-adjusted hurdle
(loss aversion). The airline wants a pricing policy that maximizes expected
revenue over the booking horizon, subject to the seats actually available.

1. STATE S_t = (t, c_t, r_t)
   t   : day index, 0 .. horizon_days-1 (t=0 is horizon_days before departure)
   c_t : seats remaining (integer, starts at capacity C, c_t >= 0)
   r_t : customers' reference price (float, a slow-moving anchor)
   S_t is a sufficient statistic: nothing before t affects what happens after,
   given (t, c_t, r_t) -- the Markov property the course decks lean on.

2. DECISION x_t = p_t, the price posted during day t.
   Feasible region X_t = [price_floor, price_ceiling] (a policy may
   additionally snap to a discrete grid).

3. EXOGENOUS INFORMATION W_{t+1} = (N_{t+1}, {WTP_i}_{i=1..N_{t+1}})
   N_{t+1} ~ Poisson(lambda_t)            number of customer arrivals on day t
   lambda_t = lambda_base * kappa if c_t/C < scarcity_threshold else lambda_base
   WTP_i ~ Lognormal(mu_t, sigma) iid      each arriving customer's WTP
   mu_t is calibrated from the real fare-vs-lead-time curve (see
   calibration.wtp_mu_at): customers who show up close to departure skew
   higher-WTP, the standard "business travelers book late" story, while
   sigma (dispersion) is held fixed across the horizon. This is where the
   "scarcity sells" narrative lives too: lambda_t jumps once inventory gets
   thin, independent of the price charged -- the price still controls who,
   of the arrivals, actually buys.

4. TRANSITION FUNCTION S^M(S_t, x_t, W_{t+1}) -> S_{t+1}
   For each of the N_{t+1} arrivals (processed one at a time, stopping early
   if c_t hits 0): customer buys iff
       WTP_i >= p_t + eta*max(p_t - r_t, 0) - gamma*max(r_t - p_t, 0)
   (pricing above the reference costs an eta-weighted penalty in the
   hurdle, pricing below it earns a gamma-weighted discount -- loss
   aversion/reference dependence). Each sale decrements c_t by 1 and adds
   p_t to the day's revenue.
   End of day: r_{t+1} = theta_ref*r_t + (1-theta_ref)*p_t   (reference
   price drifts toward whatever was actually posted), t_{t+1} = t+1.

5. OBJECTIVE
   max_pi  E[ sum_{t=0}^{horizon_days-1} sum_{i=1}^{N_{t+1}} p_t * 1{buy_i} ]
   i.e. maximize expected total revenue over the booking horizon, where the
   expectation is over the exogenous process (arrivals and WTP draws) induced
   by following policy pi = X^pi(S_t | theta). We never get to re-sell a
   seat once capacity hits 0 (c_t=0 => no further sales, by construction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ticket_pricing.calibration import Calibration, wtp_mu_at


@dataclass(frozen=True)
class State:
    t: int          # day index, 0 = horizon_days before departure
    capacity: int   # seats remaining, c_t
    ref_price: float  # reference price, r_t


@dataclass(frozen=True)
class StepResult:
    state: State          # S_{t+1}
    revenue: float         # realized revenue collected on day t
    n_arrivals: int
    n_sales: int
    done: bool             # True once t == horizon_days or capacity == 0


class Policy(Protocol):
    """X^pi(S_t | theta) -> p_t"""

    def price(self, state: State) -> float: ...


def purchase_hurdle(price: float, ref_price: float, eta: float, gamma: float) -> float:
    """The WTP threshold a customer must clear to buy at `price`, given
    reference price `ref_price`. Pricing above the reference raises the
    hurdle (eta penalty); pricing below it lowers it (gamma discount)."""
    return price + eta * max(price - ref_price, 0.0) - gamma * max(ref_price - price, 0.0)


def arrival_rate(state: State, calib: Calibration) -> float:
    """lambda_t: Poisson arrival rate for day t, bumped by kappa once the
    remaining-capacity fraction drops below the scarcity threshold."""
    frac_remaining = state.capacity / calib.capacity
    if frac_remaining < calib.scarcity_threshold:
        return calib.arrival_rate_base * calib.kappa_scarcity_multiplier
    return calib.arrival_rate_base


def draw_exogenous_info(state: State, calib: Calibration, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    """Draw W_{t+1} = (N_{t+1}, WTP draws) given only S_t and the calibration
    (never given the price -- the arrival process does not see x_t)."""
    lam = arrival_rate(state, calib)
    n = int(rng.poisson(lam))
    lead_days_remaining = calib.horizon_days - state.t
    mu_t = wtp_mu_at(calib, lead_days_remaining)
    wtp = rng.lognormal(mean=mu_t, sigma=calib.wtp_sigma, size=n) if n > 0 else np.empty(0)
    return n, wtp


def transition(
    state: State,
    price: float,
    n_arrivals: int,
    wtp_draws: np.ndarray,
    calib: Calibration,
) -> StepResult:
    """S^M(S_t, x_t, W_{t+1}) -> S_{t+1}, plus the revenue realized on day t."""
    capacity = state.capacity
    revenue = 0.0
    n_sales = 0
    hurdle = purchase_hurdle(price, state.ref_price, calib.eta_loss_aversion, calib.gamma_gain)
    for wtp in wtp_draws:
        if capacity <= 0:
            break
        if wtp >= hurdle:
            capacity -= 1
            n_sales += 1
            revenue += price

    new_ref = calib.theta_ref * state.ref_price + (1 - calib.theta_ref) * price
    new_t = state.t + 1
    done = (new_t >= calib.horizon_days) or (capacity <= 0)
    new_state = State(t=new_t, capacity=capacity, ref_price=new_ref)
    return StepResult(state=new_state, revenue=revenue, n_arrivals=n_arrivals, n_sales=n_sales, done=done)


def initial_state(calib: Calibration) -> State:
    return State(t=0, capacity=calib.capacity, ref_price=calib.wtp_median)
