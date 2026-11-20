"""
End-to-end experiment: calibrate -> tune -> evaluate -> recommend.

Produces:
  reports/results.json           tuned thetas + eval stats for both scenarios
  reports/fig_price_vs_leadtime.png   calibrated WTP curve (data-driven)
  reports/fig_revenue_dist.png        revenue distributions, base scenario
  reports/fig_price_path.png          one representative episode's price path per policy

Run from the project root:  python scripts/run_experiment.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ticket_pricing.calibration import fit_calibration, save_calibration, load_calibration, wtp_mu_at
from ticket_pricing.evaluation import evaluate, summarize, EVAL_SEEDS
from ticket_pricing.tuning import tune_baseline, tune_pfa, TUNING_SEEDS
from ticket_pricing.simulator import run_episode
from ticket_pricing.policies.baseline import FixedPricePolicy
from ticket_pricing.policies.pfa import LinearPFA
from ticket_pricing.policies.dla import DirectLookaheadPolicy

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def run_scenario(calib, label: str) -> dict:
    tb = tune_baseline(calib)
    tp = tune_pfa(calib, n_candidates=250, seed_price=tb.theta["price_level"])

    base_pol = FixedPricePolicy(price_level=tb.theta["price_level"])
    pfa_pol = LinearPFA(calib=calib, **tp.theta)
    dla_pol = DirectLookaheadPolicy(calib=calib, lookahead_days=calib.horizon_days, n_grid=25)

    results = [
        evaluate(base_pol, "baseline", calib),
        evaluate(pfa_pol, "pfa", calib),
        evaluate(dla_pol, "dla", calib),
    ]
    print(f"\n=== scenario: {label} (arrival_rate_base={calib.arrival_rate_base:.3f}, "
          f"E[arrivals]={calib.arrival_rate_base*calib.horizon_days:.1f}, capacity={calib.capacity}) ===")
    print(summarize(results))

    return {
        "label": label,
        "arrival_rate_base": calib.arrival_rate_base,
        "tuning": {"baseline": tb.theta, "pfa": tp.theta},
        "eval": {
            r.name: {"mean": r.mean, "std": r.std, "p05": r.p05, "p95": r.p95,
                      "ci95": [r.ci95_low, r.ci95_high], "n": r.n}
            for r in results
        },
        "_policies": {"baseline": base_pol, "pfa": pfa_pol, "dla": dla_pol},
        "_results": results,
    }


def plot_wtp_curve(calib):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lead_days = np.linspace(0, calib.horizon_days, 200)
    mu_t = np.array([wtp_mu_at(calib, ld) for ld in lead_days])
    median_wtp = np.exp(mu_t)
    ax.plot(calib.horizon_days - lead_days, median_wtp, color="#2563eb", lw=2,
            label="fitted median WTP (used by the simulator)")
    bucket_lead = np.array(calib.price_curve_lead_days)
    ax.scatter(calib.horizon_days - bucket_lead, calib.price_curve_mean_fare,
                color="#ef4444", zorder=5, label="raw observed mean fare (noisy bucket averages)")
    ax.set_xlabel("days since booking opened (t)")
    ax.set_ylabel("USD")
    ax.set_title(f"Calibrated WTP vs. time -- {calib.route}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_price_vs_leadtime.png", dpi=130)
    plt.close(fig)


def plot_revenue_dist(scenario: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in scenario["_results"]:
        ax.hist(r.revenues, bins=30, alpha=0.5, label=f"{r.name} (mean={r.mean:.0f})")
    ax.set_xlabel("total revenue per episode (USD)")
    ax.set_ylabel("count")
    ax.set_title(f"Revenue distribution, {scenario['_results'][0].n} eval scenarios\n{scenario['label']}", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_revenue_dist.png", dpi=130)
    plt.close(fig)


def plot_price_path(calib, scenario: dict, seed: int = 200000):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, pol in scenario["_policies"].items():
        log = run_episode(pol, calib, seed=seed)
        ax.step(range(len(log.prices)), log.prices, where="post", label=name)
    ax.set_xlabel("day t")
    ax.set_ylabel("posted price (USD)")
    ax.set_title(f"Price path, one representative scenario (seed={seed})\n{scenario['label']}", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_price_path.png", dpi=130)
    plt.close(fig)


def main():
    calib = fit_calibration()
    save_calibration(calib)

    base_scenario = run_scenario(calib, "real-data calibration (ATL-BOS, moderate demand)")
    stress_calib = dataclasses.replace(calib, arrival_rate_base=calib.capacity / calib.horizon_days * 3.0)
    stress_scenario = run_scenario(stress_calib, "high-demand sensitivity check (demand ~3x capacity)")

    plot_wtp_curve(calib)
    plot_revenue_dist(base_scenario)
    plot_price_path(calib, base_scenario)

    out = {
        "calibration": {
            "route": calib.route, "n_obs": calib.n_obs, "capacity": calib.capacity,
            "horizon_days": calib.horizon_days, "wtp_sigma": calib.wtp_sigma,
            "wtp_mu_poly_coeffs": calib.wtp_mu_poly_coeffs,
        },
        "tuning_seeds": [TUNING_SEEDS[0], TUNING_SEEDS[-1], len(TUNING_SEEDS)],
        "eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1], len(EVAL_SEEDS)],
        "scenarios": [
            {k: v for k, v in s.items() if not k.startswith("_")}
            for s in (base_scenario, stress_scenario)
        ],
    }
    with open(REPORTS / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {REPORTS / 'results.json'} and figures to {REPORTS}/")


if __name__ == "__main__":
    main()
