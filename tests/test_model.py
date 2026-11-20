import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from ticket_pricing.calibration import load_calibration, wtp_mu_at
from ticket_pricing.model import State, initial_state, purchase_hurdle, transition, arrival_rate
from ticket_pricing.simulator import TicketPricingEnv, run_episode, run_batch
from ticket_pricing.policies.baseline import FixedPricePolicy
from ticket_pricing.policies.pfa import LinearPFA
from ticket_pricing.policies.dla import DirectLookaheadPolicy


@pytest.fixture(scope="module")
def calib():
    return load_calibration()


def test_initial_state(calib):
    s = initial_state(calib)
    assert s.t == 0
    assert s.capacity == calib.capacity
    assert s.ref_price > 0


def test_purchase_hurdle_monotonic(calib):
    # pricing above the reference should only ever raise the hurdle
    ref = 300.0
    h_low = purchase_hurdle(250, ref, calib.eta_loss_aversion, calib.gamma_gain)
    h_mid = purchase_hurdle(300, ref, calib.eta_loss_aversion, calib.gamma_gain)
    h_high = purchase_hurdle(400, ref, calib.eta_loss_aversion, calib.gamma_gain)
    assert h_low < h_mid < h_high
    # at price == ref, hurdle == price exactly (no penalty, no discount)
    assert purchase_hurdle(ref, ref, calib.eta_loss_aversion, calib.gamma_gain) == pytest.approx(ref)


def test_capacity_never_negative(calib):
    state = State(t=10, capacity=1, ref_price=300.0)
    # 5 very-low-hurdle arrivals trying to buy against only 1 seat left
    wtp = np.array([10_000, 10_000, 10_000, 10_000, 10_000], dtype=float)
    result = transition(state, price=1.0, n_arrivals=5, wtp_draws=wtp, calib=calib)
    assert result.state.capacity == 0
    assert result.n_sales == 1  # only one seat was available


def test_episode_terminates(calib):
    pol = FixedPricePolicy(price_level=calib.wtp_median)
    log = run_episode(pol, calib, seed=0)
    assert log.seats_remaining >= 0
    assert log.seats_sold + log.seats_remaining == calib.capacity
    assert log.total_revenue >= 0


def test_crn_determinism(calib):
    pol = FixedPricePolicy(price_level=calib.wtp_median)
    log1 = run_episode(pol, calib, seed=42)
    log2 = run_episode(pol, calib, seed=42)
    assert log1.total_revenue == log2.total_revenue
    assert log1.prices == log2.prices


def test_crn_cross_policy_same_seed_same_arrivals(calib):
    # two different policies facing the SAME seed should see the same
    # scenario "shape" (this is what makes evaluation.py's comparisons fair):
    # a cheap policy should never sell fewer seats than a pricier one
    # when both face the identical draw sequence (monotonicity sanity check)
    cheap = FixedPricePolicy(price_level=calib.price_floor)
    pricey = FixedPricePolicy(price_level=calib.price_ceiling)
    log_cheap = run_episode(cheap, calib, seed=5)
    log_pricey = run_episode(pricey, calib, seed=5)
    assert log_cheap.seats_sold >= log_pricey.seats_sold


def test_pfa_price_within_bounds(calib):
    pol = LinearPFA(theta_a=calib.price_floor, theta_b=100, theta_c=100, calib=calib)
    for t in [0, 30, 59]:
        for cap in [calib.capacity, calib.capacity // 2, 1]:
            s = State(t=t, capacity=cap, ref_price=calib.wtp_median)
            p = pol.price(s)
            assert calib.price_floor <= p <= calib.price_ceiling


def test_dla_price_within_bounds(calib):
    pol = DirectLookaheadPolicy(calib=calib, lookahead_days=10, n_grid=8)
    s = State(t=5, capacity=5, ref_price=calib.wtp_median)
    p = pol.price(s)
    assert calib.price_floor <= p <= calib.price_ceiling


def test_wtp_mu_at_clips_outside_range(calib):
    far_future = wtp_mu_at(calib, calib.horizon_days + 1000)
    at_edge = wtp_mu_at(calib, calib.price_curve_lead_days[-1])
    assert far_future == pytest.approx(at_edge)


def test_run_batch_matches_run_episode(calib):
    pol = FixedPricePolicy(price_level=calib.wtp_median)
    seeds = [1, 2, 3]
    batch = run_batch(pol, calib, seeds)
    individual = [run_episode(pol, calib, s).total_revenue for s in seeds]
    assert list(batch) == pytest.approx(individual)


def test_arrival_rate_jumps_under_scarcity(calib):
    plenty = State(t=0, capacity=calib.capacity, ref_price=calib.wtp_median)
    scarce = State(t=0, capacity=0, ref_price=calib.wtp_median)
    assert arrival_rate(scarce, calib) > arrival_rate(plenty, calib)
