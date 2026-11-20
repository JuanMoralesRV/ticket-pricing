"""Baseline policy: a single constant price for the whole booking horizon.

This is the "buy and hold" / "greedy" yardstick the lecture slides always put
next to a tuned PFA or CFA (see "What tuning bought us" in 008_PFA_and_CFA):
every other policy has to beat this to justify its own complexity.
"""
from __future__ import annotations

from dataclasses import dataclass

from ticket_pricing.model import State


@dataclass
class FixedPricePolicy:
    price_level: float  # theta: the one parameter of this policy class

    def price(self, state: State) -> float:
        return self.price_level
