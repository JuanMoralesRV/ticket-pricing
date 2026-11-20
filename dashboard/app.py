"""
Streamlit dashboard for Project B: Dynamic ticket pricing.

Run with:  streamlit run dashboard/app.py   (from the project root)

Lets you:
  - inspect the calibration (WTP-vs-leadtime curve fit from real fare data)
  - tune / pick theta for the baseline and PFA policies interactively
  - compare baseline vs. PFA vs. DLA on a common-random-number batch of
    scenarios, with mean / spread / CI
  - watch a single representative episode's price path and capacity
    depletion for every policy side by side
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from ticket_pricing.calibration import load_calibration, wtp_mu_at, Calibration
from ticket_pricing.evaluation import evaluate, EVAL_SEEDS
from ticket_pricing.simulator import run_episode
from ticket_pricing.policies.baseline import FixedPricePolicy
from ticket_pricing.policies.pfa import LinearPFA
from ticket_pricing.policies.dla import DirectLookaheadPolicy

st.set_page_config(page_title="Dynamic Ticket Pricing", layout="wide")

CALIB_PATH = ROOT / "data" / "calibration.json"

if not CALIB_PATH.exists():
    st.error("No calibration found. Run `python src/ticket_pricing/calibration.py` first.")
    st.stop()

base_calib = load_calibration(CALIB_PATH)

st.title("Dynamic ticket pricing -- Project B")
st.caption(
    f"Powell model calibrated on {base_calib.n_obs:,} real fare observations "
    f"({base_calib.route}, Kaggle 'Flight Prices' dataset)."
)

with st.sidebar:
    st.header("Scenario")
    demand_mult = st.slider(
        "Demand pressure (expected arrivals / capacity)", 0.5, 4.0, 1.4, 0.1,
        help="arrival_rate_base = capacity/horizon * this multiplier. The real-data "
             "calibration sits at ~1.4x; the report's sensitivity check uses 3.0x.",
    )
    calib = dataclasses.replace(
        base_calib, arrival_rate_base=base_calib.capacity / base_calib.horizon_days * demand_mult
    )
    st.metric("E[arrivals] over horizon", f"{calib.arrival_rate_base*calib.horizon_days:.1f}")
    st.metric("Capacity", calib.capacity)

    st.divider()
    st.header("Baseline policy theta")
    base_price = st.slider("Fixed price (USD)", float(calib.price_floor), float(calib.price_ceiling),
                            float(calib.wtp_median), 1.0)

    st.divider()
    st.header("PFA policy theta")
    span = calib.price_ceiling - calib.price_floor
    theta_a = st.slider("theta_a (base anchor, USD)", float(calib.price_floor), float(calib.price_ceiling),
                         float(base_price), 1.0)
    theta_b = st.slider("theta_b (scarcity markup, USD)", 0.0, float(span), 0.0, 1.0)
    theta_c = st.slider("theta_c (time markup, USD)", 0.0, float(span), 0.0, 1.0)

    st.divider()
    st.header("DLA policy settings")
    n_grid = st.slider("Price-grid resolution", 5, 40, 25, 1)
    n_eval = st.slider("Evaluation scenarios (CRN)", 50, 600, 300, 50)

base_pol = FixedPricePolicy(price_level=base_price)
pfa_pol = LinearPFA(theta_a=theta_a, theta_b=theta_b, theta_c=theta_c, calib=calib)
dla_pol = DirectLookaheadPolicy(calib=calib, lookahead_days=calib.horizon_days, n_grid=n_grid)
policies = {"baseline": base_pol, "pfa": pfa_pol, "dla": dla_pol}

tab1, tab2, tab3 = st.tabs(["Calibration", "Evaluation (CRN)", "Episode trace"])

with tab1:
    st.subheader("Calibrated WTP vs. days since booking opened")
    lead_days = np.linspace(0, calib.horizon_days, 200)
    mu_t = np.array([wtp_mu_at(calib, ld) for ld in lead_days])
    df_curve = pd.DataFrame({
        "t (days since booking opened)": calib.horizon_days - lead_days,
        "median WTP (USD)": np.exp(mu_t),
    }).set_index("t (days since booking opened)")
    st.line_chart(df_curve)
    st.caption(
        "Fit: degree-2 regression of log(totalFare) on days-to-departure, on "
        f"{calib.n_obs:,} real itinerary rows. sigma (dispersion) = {calib.wtp_sigma:.3f}."
    )
    raw = pd.DataFrame({
        "lead_days (bucket right edge)": calib.price_curve_lead_days,
        "observed mean fare (USD)": calib.price_curve_mean_fare,
    })
    st.dataframe(raw, hide_index=True)

with tab2:
    st.subheader(f"Out-of-sample evaluation on {n_eval} common-random-number scenarios")
    seeds = EVAL_SEEDS[:n_eval]
    rows = []
    revenue_by_policy = {}
    for name, pol in policies.items():
        r = evaluate(pol, name, calib, seeds=seeds)
        revenue_by_policy[name] = r.revenues
        rows.append({
            "policy": name, "mean": round(r.mean, 1), "std": round(r.std, 1),
            "p05": round(r.p05, 1), "p95": round(r.p95, 1),
            "95% CI": f"[{r.ci95_low:.1f}, {r.ci95_high:.1f}]", "n": r.n,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.bar_chart(pd.DataFrame({k: [v.mean()] for k, v in revenue_by_policy.items()}, index=["mean revenue"]).T)
    st.caption(
        "Same seed list for every policy (common random numbers): differences above are "
        "attributable to the policy, not to which customers happened to show up."
    )

with tab3:
    st.subheader("One representative scenario, side by side")
    seed = st.number_input("Scenario seed", min_value=0, value=EVAL_SEEDS[0], step=1)
    cols = st.columns(2)
    price_df = {}
    cap_df = {}
    summary_rows = []
    for name, pol in policies.items():
        log = run_episode(pol, calib, seed=seed)
        price_df[name] = pd.Series(log.prices)
        cap_df[name] = pd.Series(log.capacities)
        summary_rows.append({"policy": name, "revenue": round(log.total_revenue, 1),
                              "seats_sold": log.seats_sold, "seats_remaining": log.seats_remaining})
    with cols[0]:
        st.markdown("**Posted price per day**")
        st.line_chart(pd.DataFrame(price_df))
    with cols[1]:
        st.markdown("**Capacity remaining**")
        st.line_chart(pd.DataFrame(cap_df))
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
