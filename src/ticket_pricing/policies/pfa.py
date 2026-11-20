"""Policy Function Approximation: a parametrized function that maps the
state directly to a price, with no optimization solved at decision time --
exactly the PFA definition in 007_Policy_Classes ("the shape of a good rule
is known; the decision is low-dimensional and must be fast and explainable").

X^PFA(S_t | theta) = clip( theta_a + theta_b * (1 - c_t/C) + theta_c * (t/T), floor, ceiling )

theta_a : base price anchor
theta_b : scarcity markup -- how much price rises as remaining capacity c_t/C shrinks
theta_c : time markup -- how much price rises as departure (t/T) approaches
Both markup terms are in [0, 1] by construction, so theta_b, theta_c are
directly in price units (USD added at full scarcity / at departure).
"""
from __future__ import annotations

from dataclasses import dataclass

from ticket_pricing.calibration import Calibration
from ticket_pricing.model import State


@dataclass
class LinearPFA:
    theta_a: float
    theta_b: float
    theta_c: float
    calib: Calibration

    def price(self, state: State) -> float:
        scarcity_term = 1.0 - state.capacity / self.calib.capacity
        time_term = state.t / self.calib.horizon_days
        p = self.theta_a + self.theta_b * scarcity_term + self.theta_c * time_term
        return float(min(max(p, self.calib.price_floor), self.calib.price_ceiling))
