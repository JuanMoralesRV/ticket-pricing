"""Streamlit dashboard for Project B. Run with streamlit run dashboard/app.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ticket_pricing.concierto import (
    CAL, FINAL, SELECT, Evento, Plan, estudio, intervalo, precio_regla, sensibilidad, trio,
)

POLICIES = ("baseline", "pfa", "dla")
DISPLAY = ("Fixed price", "Sales rule (PFA)", "Daily lookahead (DLA)")
st.set_page_config(page_title="Dynamic Ticket Pricing", layout="wide")


@st.cache_resource(max_entries=10, show_spinner=False)
def fitted_run(concert):
    return estudio(concert)


@st.cache_data(show_spinner=False)
def observed_fares():
    data = pd.read_csv(ROOT / "data" / "ATL_BOS.csv", parse_dates=["searchDate", "flightDate"])
    data["days"] = (data.flightDate - data.searchDate).dt.days
    data = data.query("1 <= days <= 60 and totalFare > 0")
    fit = np.polyfit(data.days, np.log(data.totalFare), 2)
    days = np.linspace(1, 60, 200)
    curve = pd.DataFrame({
        "t (days since booking opened)": 60 - days,
        "fitted fare (USD)": np.exp(np.polyval(fit, days)),
    }).set_index("t (days since booking opened)")
    edges = [0, 3, 7, 14, 21, 30, 45, 60]
    buckets = pd.cut(data.days, edges, labels=edges[1:])
    means = data.groupby(buckets, observed=True).totalFare.mean()
    table = pd.DataFrame({
        "lead_days (bucket right edge)": means.index.astype(int),
        "observed mean fare (USD)": means.to_numpy().round(2),
    })
    return curve, table


@st.cache_resource(max_entries=6, show_spinner=False)
def manual_run(concert, fixed_price, coefficients, resolution):
    plan = Plan(concert, n_stock=resolution)
    selection = trio(concert, SELECT, fixed_price, coefficients, plan)
    choice = 0
    for challenger in (1, 2):
        difference = selection[challenger]["ingresos"] - selection[choice]["ingresos"]
        if intervalo(difference)[0] > 0:
            choice = challenger
    final = trio(concert, FINAL, fixed_price, coefficients, plan, trace=True)
    return final, plan, choice


st.title("Dynamic ticket pricing -- Project B")
st.caption(
    f"Concert pricing with the Powell model. Arrival timing estimated from {CAL['n_rows']:,} "
    "real fare and seat-availability observations (ATL–BOS nonstop, Kaggle 'Flight Prices')."
)

with st.sidebar:
    scenario_panel = st.container()
    st.divider()
    baseline_panel = st.container()
    st.divider()
    pfa_panel = st.container()
    st.divider()
    dla_panel = st.container()
    st.divider()
    with st.expander("Concert & demand assumptions"):
        capacity = st.number_input("Venue capacity", 100, 8000, 2000, 100)
        horizon = st.number_input("Selling window (days)", 7, 90, 60)
        floor = st.number_input("Minimum price (USD)", 5, 500, 40, 5)
        ceiling = st.number_input("Maximum price (USD)", 10, 600, 200, 5)
        step = st.selectbox("Price step (USD)", [5, 10, 20, 25, 50], index=1)
        initial_reference = st.number_input("Initial reference price (USD)", 5., 600., 100., 5.)
        median_wtp = st.number_input("Median WTP (USD)", 10., 500., 105., 5.)
        dispersion = st.slider("WTP log dispersion", .15, .90, .42, .01)
        memory = st.slider("Reference memory", 0., 1., .75, .05)
        penalty = st.slider("Penalty above reference", 0., 2., .80, .05)
        reward = st.slider("Reward below reference", 0., 1., .30, .05)
        scarcity = st.slider("Scarcity multiplier", 1., 2., 1.35, .05)

with scenario_panel:
    st.header("Scenario")
    demand = st.slider(
        "Demand pressure (expected arrivals / capacity)", 0.5, 4.0, 2.2, 0.1,
        help="Expected potential buyers per seat, before the scarcity effect.",
    )
    try:
        concert = Evento(capacidad=capacity, dias=horizon, minimo=floor, maximo=ceiling,
                         paso=step, demanda=demand, mediana=median_wtp, sigma=dispersion,
                         referencia=initial_reference, theta=memory, eta=penalty, gamma=reward, kappa=scarcity)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    st.metric("E[arrivals] over horizon", f"{demand * capacity:.1f}", help="Base arrivals, before scarcity.")
    st.metric("Capacity", capacity)

study = fitted_run(concert)
scenario_id = repr(concert)
with baseline_panel:
    st.header("Baseline policy theta")
    fixed = st.slider("Fixed price (USD)", floor, ceiling, int(study["fijo"]), step, key="fixed" + scenario_id)
with pfa_panel:
    st.header("PFA policy theta")
    a, b, c, d = map(float, study["regla"])
    span = float(ceiling - floor)
    anchor = st.slider("theta_a (base anchor, USD)", floor, ceiling, int(a), step, key="anchor" + scenario_id)
    pace = st.slider("theta_b (sales pace markup, USD)", 0., span, b, .1, key="pace" + scenario_id)
    time = st.slider("theta_c (time markup, USD)", -span, span, c, .1, key="time" + scenario_id)
    recent = st.slider("theta_d (recent sales markup, USD)", 0., span, d, .1, key="recent" + scenario_id)
with dla_panel:
    st.header("DLA policy settings")
    resolution = st.slider("Inventory-grid resolution", 41, 121, 81, 10)
    n_eval = st.slider("Evaluation scenarios (CRN)", 50, FINAL[1], FINAL[1], 50)

parameters = (anchor, pace, time, recent)
manual = fixed != study["fijo"] or not np.allclose(parameters, study["regla"]) or resolution != 81
if manual:
    results, plan, choice = manual_run(concert, fixed, parameters, resolution)
else:
    results, plan, choice = study["resultados"], study["plan"], study["mejor"]

calibration, evaluation, episode = st.tabs(["Calibration", "Evaluation (CRN)", "Episode trace"])

with calibration:
    st.subheader("Calibrated fare vs. days since booking opened")
    curve, fares = observed_fares()
    st.line_chart(curve)
    st.caption("Fit: degree-2 regression of log(totalFare) on days to departure. The curve describes published airfares.")
    st.dataframe(fares, hide_index=True)
    with st.expander("Concert arrival-rate calibration"):
        arrivals = pd.DataFrame({"Expected arrivals": concert.llegadas},
                                index=pd.Index(range(1, horizon + 1), name="Day"))
        st.line_chart(arrivals)
        lower, upper = CAL["ratio_ci"]
        st.caption(f"{CAL['n_pairs']:,} paired observations. End/start ratio {CAL['end_start_ratio']:.2f}× "
                   f"(95% CI {lower:.2f}–{upper:.2f}). Concert demand volume and WTP are assumptions.")
        buckets = pd.DataFrame(CAL["buckets"])
        st.dataframe(pd.DataFrame({
            "days to departure": [f"{10*k+1}–{10*(k+1)}" for k in buckets.bucket],
            "pairs": buckets.n,
            "mean prior seats": buckets.seats.round(2),
            "mean prior fare (USD)": buckets.fare.round(2),
            "mean inventory decrease": buckets["drop"].round(2),
        }), hide_index=True)

with evaluation:
    st.subheader(f"Out-of-sample evaluation on {n_eval} common-random-number scenarios")
    opening_price = results[choice]["precios"][0, 0]
    label = "Manual comparison" if manual or n_eval != FINAL[1] else "Recommended policy"
    st.caption(f"{label}: {DISPLAY[choice]} · Opening price: USD {opening_price:,.0f}.")
    revenue = {}
    rows = []
    for name, result in zip(POLICIES, results):
        x = result["ingresos"][:n_eval]
        lower, upper = intervalo(x)
        rows.append({
            "policy": name, "mean": round(x.mean(), 1), "std": round(x.std(ddof=1), 1),
            "p05": round(np.percentile(x, 5), 1), "p95": round(np.percentile(x, 95), 1),
            "95% CI": f"[{lower:.1f}, {upper:.1f}]", "n": n_eval,
            "sell-out": f"{100 * result['lleno'][:n_eval].mean():.1f}%",
        })
        revenue[name] = x
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.bar_chart(pd.DataFrame({"mean revenue (USD)": [x.mean() for x in revenue.values()]}, index=POLICIES))

    with st.expander("Revenue distribution and paired differences"):
        edges = np.linspace(min(x.min() for x in revenue.values()) - .5,
                            max(x.max() for x in revenue.values()) + .5, 25)
        histogram = pd.DataFrame(
            {name: 100 * np.histogram(x, edges)[0] / len(x) for name, x in revenue.items()},
            index=pd.Index((edges[:-1] + edges[1:]) / 2, name="Revenue (USD), bin midpoint"),
        )
        st.bar_chart(histogram, stack=False, y_label="Scenarios (%)")
        differences = []
        for name in POLICIES[1:]:
            diff = revenue[name] - revenue["baseline"]
            lower, upper = intervalo(diff)
            differences.append({"policy": name, "gain vs. baseline": round(diff.mean(), 1),
                                "paired 95% CI": f"[{lower:.1f}, {upper:.1f}]"})
        st.dataframe(pd.DataFrame(differences), hide_index=True)

    with st.expander("Reference-price sensitivity · re-tuned policies"):
        variants = []
        for factor, setting in [(0, "Off"), (.5, "Half"), (1, "Configured"), (1.5, "Stronger")]:
            variant = fitted_run(sensibilidad(concert, factor))
            winner = variant["mejor"]
            best = variant["resultados"][winner]
            variants.append({
                "reference effect": setting, "selected policy": POLICIES[winner],
                "opening price": best["precios"][0, 0],
                "mean revenue": round(best["ingresos"].mean(), 1),
                "sell-out": f"{100 * best['lleno'].mean():.1f}%",
            })
        st.dataframe(pd.DataFrame(variants), hide_index=True)
        st.caption("600 common final scenarios per row. Each setting uses its own tuned policies.")

with episode:
    st.subheader("One representative scenario, side by side")
    scenario_number = st.number_input("Scenario number", min_value=1, max_value=FINAL[1], value=1, step=1)
    index = scenario_number - 1
    prices, inventories, summary = {}, {}, []
    for name, result in zip(POLICIES, results):
        prices[name] = pd.Series(result["precios"][:, index], index=range(1, horizon + 1))
        left = int(result["restantes"][index])
        inventories[name] = pd.Series(np.r_[result["stock"][:, index], left], index=range(1, horizon + 2))
        summary.append({"policy": name, "revenue": round(result["ingresos"][index], 1),
                        "seats_sold": capacity - left, "seats_remaining": left})
    price_column, capacity_column = st.columns(2)
    with price_column:
        st.markdown("**Posted price per day**")
        st.line_chart(pd.DataFrame(prices))
    with capacity_column:
        st.markdown("**Capacity remaining**")
        st.line_chart(pd.DataFrame(inventories))
    st.dataframe(pd.DataFrame(summary), hide_index=True)

    with st.expander("Today's recommended price"):
        left_column, right_column = st.columns(2)
        day = left_column.number_input("Day of sale", 1, horizon, 1, key=f"day{horizon}")
        stock = right_column.number_input("Available tickets", 0, capacity, capacity, key=f"stock{capacity}")
        ref = left_column.number_input("Reference price (USD)", float(floor), float(ceiling),
                                       float(initial_reference), 1., key=f"ref{floor}_{ceiling}_{initial_reference}")
        sales = right_column.number_input("Recent sales (tickets/day)", 0., float(capacity),
                                          round(capacity / horizon, 2), 1., key=f"rate{capacity}_{horizon}")
        if stock:
            elapsed_demand = np.r_[0, concert.llegadas.cumsum()] / concert.llegadas.sum()
            quotes = [fixed, precio_regla(concert, day - 1, stock, sales, parameters, elapsed_demand),
                      plan.precio(day - 1, stock, ref)]
            st.metric(DISPLAY[choice], f"USD {quotes[choice]:,.0f}")
        else:
            st.write("Sold out")
