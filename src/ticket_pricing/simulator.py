"""
A gym/CartPole-style step() simulator wrapping the Powell model in model.py,
plus a run_episode / run_batch harness that supports common random numbers
(CRN): the same seed always produces the same sequence of arrivals and WTP
draws, so two different policies facing seed k face an identical customer
stream and differences in revenue are attributable to the policy, not to
sampling noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ticket_pricing.calibration import Calibration
from ticket_pricing.model import Policy, State, draw_exogenous_info, initial_state, transition


class TicketPricingEnv:
    """reset() -> State; step(price) -> (State, revenue, done, info)."""

    def __init__(self, calib: Calibration, seed: int | None = None):
        self.calib = calib
        self.rng = np.random.default_rng(seed)
        self.state: State = initial_state(calib)

    def reset(self, seed: int | None = None) -> State:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = initial_state(self.calib)
        return self.state

    def step(self, price: float):
        n, wtp = draw_exogenous_info(self.state, self.calib, self.rng)
        result = transition(self.state, price, n, wtp, self.calib)
        self.state = result.state
        info = {"n_arrivals": result.n_arrivals, "n_sales": result.n_sales}
        return result.state, result.revenue, result.done, info


@dataclass
class EpisodeLog:
    total_revenue: float
    seats_sold: int
    seats_remaining: int
    prices: list
    revenues: list
    capacities: list


def run_episode(policy: Policy, calib: Calibration, seed: int) -> EpisodeLog:
    env = TicketPricingEnv(calib, seed=seed)
    state = env.reset()
    prices, revenues, capacities = [], [], [0] * 0
    capacities = [state.capacity]
    total_revenue = 0.0
    seats_sold = 0
    done = False
    while not done:
        price = policy.price(state)
        state, revenue, done, info = env.step(price)
        prices.append(price)
        revenues.append(revenue)
        capacities.append(state.capacity)
        total_revenue += revenue
        seats_sold += info["n_sales"]
    return EpisodeLog(
        total_revenue=total_revenue,
        seats_sold=seats_sold,
        seats_remaining=state.capacity,
        prices=prices,
        revenues=revenues,
        capacities=capacities,
    )


def run_batch(policy: Policy, calib: Calibration, seeds: list[int]) -> np.ndarray:
    """Return an array of total revenue, one per seed. Pass the SAME `seeds`
    list to every policy you want to compare -- that is what makes it CRN."""
    return np.array([run_episode(policy, calib, seed).total_revenue for seed in seeds])
