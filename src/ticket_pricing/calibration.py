"""
Calibration of the dynamic ticket pricing model from real fare data.

Data source: Kaggle "Flight Prices" dataset (dilwong/flightprices), a public
dump of one-way US domestic itinerary searches scraped daily in April 2022
for flights departing April-June 2022 (searchDate in 2022-04-16..04-18,
flightDate up to 60 days later). We work off a ~1M-row sample of the raw
31 GB file (data/itineraries_sample.csv) pulled directly off the file's
compressed byte stream, since downloading the full file is impractical.

What we calibrate FROM the data (route: Atlanta-Boston, nonstop, main cabin):
  - the willingness-to-pay (WTP) distribution: fit a lognormal to the observed
    totalFare, which is a reasonable lower-bound proxy for WTP (a ticket only
    sells if WTP >= price, so prices people actually paid sit inside the WTP
    distribution's support, not above it -- the fit is a modeling choice, not
    an exact recovery of WTP, and is documented as such in the report).
  - a relative, lead-time-dependent price curve, as a sanity-check reference
    (not fed directly into the simulator, which generates its own reference
    price endogenously through the loss-aversion recursion).
  - the capacity scale: the dataset caps displayed seatsRemaining at 10, which
    we read as "one fare bucket / booking class allotment" rather than a full
    aircraft, and use directly as the simulated capacity C.

What we ASSUME (no field in this dataset identifies them; chosen from the
project narrative / standard revenue-management ranges, and explicitly
surfaced here so they can be sensitivity-tested):
  - reference-price smoothing theta_ref, loss-aversion eta, gain gamma,
  - the scarcity arrival multiplier kappa (the <10% capacity trigger itself
    is specified by the assignment, kappa's magnitude is not).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_SAMPLE = DATA_DIR / "itineraries_sample.csv"
CALIBRATION_OUT = DATA_DIR / "calibration.json"


@dataclass
class Calibration:
    route: str
    n_obs: int
    horizon_days: int
    capacity: int
    wtp_mu: float          # lognormal location, fit on log(totalFare)
    wtp_sigma: float        # lognormal scale
    wtp_mean: float
    wtp_median: float
    price_floor: float      # p05 of observed fares -> lower sane bound for price grid
    price_ceiling: float    # p95 of observed fares -> upper sane bound for price grid
    price_curve_lead_days: list    # bucket right-edges, near -> far departure (for reporting/plots)
    price_curve_mean_fare: list     # raw (noisy) observed bucket means, for reporting/plots
    wtp_mu_poly_coeffs: list        # degree-2 fit of log(totalFare) ~ leadDays, highest power first
    theta_ref: float         # reference-price smoothing (assumed)
    eta_loss_aversion: float  # penalty for pricing above reference (assumed)
    gamma_gain: float         # reward for pricing below reference (assumed)
    scarcity_threshold: float  # capacity fraction that triggers kappa (spec)
    kappa_scarcity_multiplier: float  # arrival-rate multiplier under scarcity (assumed)
    arrival_rate_base: float  # customers/day at full capacity, far from departure


def fit_calibration(
    origin: str = "ATL",
    destination: str = "BOS",
    nonstop_only: bool = True,
    main_cabin_only: bool = True,
    capacity: int = 10,
    horizon_days: int = 60,
    theta_ref: float = 0.7,
    eta_loss_aversion: float = 1.5,
    gamma_gain: float = 0.5,
    scarcity_threshold: float = 0.10,
    kappa_scarcity_multiplier: float = 1.3,
) -> Calibration:
    df = pd.read_csv(
        RAW_SAMPLE,
        usecols=[
            "searchDate", "flightDate", "startingAirport", "destinationAirport",
            "isNonStop", "isBasicEconomy", "totalFare", "seatsRemaining",
        ],
        parse_dates=["searchDate", "flightDate"],
    )
    df["leadDays"] = (df.flightDate - df.searchDate).dt.days

    mask = (df.startingAirport == origin) & (df.destinationAirport == destination)
    if nonstop_only:
        mask &= df.isNonStop
    if main_cabin_only:
        mask &= ~df.isBasicEconomy
    sub = df.loc[mask].copy()
    if len(sub) < 200:
        raise ValueError(f"Too few observations ({len(sub)}) for {origin}-{destination}; pick a busier route.")

    log_fare = np.log(sub.totalFare)

    # Smooth, data-driven time-varying WTP location: a degree-2 weighted
    # least-squares fit of log(totalFare) on leadDays, using every row (not
    # 7-bucket averages, which the slides warn are noisy with small counts
    # -- our 4-7 day bucket is exactly such a case). Since log(totalFare) ~
    # Normal(mu_t, sigma) under the lognormal model, the regression's fitted
    # value AT a given leadDays IS mu_t directly, no inversion needed. This
    # recovers the standard revenue-management story (late bookers skew
    # higher-WTP) straight from the data instead of assuming it.
    poly_coeffs = np.polyfit(sub.leadDays, log_fare, deg=2)
    pred = np.polyval(poly_coeffs, sub.leadDays)
    sigma = float((log_fare - pred).std())  # residual dispersion around the fitted curve
    mu = float(np.polyval(poly_coeffs, 0))   # mu at departure (leadDays=0), for headline reporting

    bins = [0, 3, 7, 14, 21, 30, 45, max(46, horizon_days)]
    sub["bucket"] = pd.cut(sub.leadDays, bins=bins)
    curve = sub.groupby("bucket", observed=True).totalFare.mean().dropna()
    bucket_right_edges = [int(iv.right) for iv in curve.index]
    bucket_means = [float(v) for v in curve.values]

    arrival_rate_base = capacity / horizon_days * 1.4  # see docstring note below

    calib = Calibration(
        route=f"{origin}-{destination} {'nonstop' if nonstop_only else 'any'}, "
              f"{'main cabin' if main_cabin_only else 'all fares'}",
        n_obs=int(len(sub)),
        horizon_days=horizon_days,
        capacity=capacity,
        wtp_mu=mu,
        wtp_sigma=sigma,
        wtp_mean=float(np.exp(mu + sigma**2 / 2)),
        wtp_median=float(np.exp(mu)),
        price_floor=float(sub.totalFare.quantile(0.05)),
        price_ceiling=float(sub.totalFare.quantile(0.95)),
        price_curve_lead_days=bucket_right_edges,
        price_curve_mean_fare=bucket_means,
        wtp_mu_poly_coeffs=[float(c) for c in poly_coeffs],
        theta_ref=theta_ref,
        eta_loss_aversion=eta_loss_aversion,
        gamma_gain=gamma_gain,
        scarcity_threshold=scarcity_threshold,
        kappa_scarcity_multiplier=kappa_scarcity_multiplier,
        arrival_rate_base=arrival_rate_base,
    )
    return calib


def save_calibration(calib: Calibration, path: Path = CALIBRATION_OUT) -> None:
    path.write_text(json.dumps(asdict(calib), indent=2))


def load_calibration(path: Path = CALIBRATION_OUT) -> Calibration:
    data = json.loads(path.read_text())
    return Calibration(**data)


def wtp_mu_at(calib: Calibration, lead_days_remaining: float) -> float:
    """Time-varying WTP location: evaluate the calibrated degree-2 fit of
    log(totalFare) on leadDays. lead_days_remaining = 0 is departure day,
    = horizon_days is the day booking opens. Clipped to the observed
    leadDays range (0..60 for this route) to avoid the polynomial running
    away outside the data it was fit on. Plain-float arithmetic (no numpy
    call overhead) since this runs in the simulator's hot per-step loop."""
    ld = lead_days_remaining
    if ld < 0:
        ld = 0.0
    elif ld > calib.price_curve_lead_days[-1]:
        ld = float(calib.price_curve_lead_days[-1])
    c2, c1, c0 = calib.wtp_mu_poly_coeffs
    return c2 * ld * ld + c1 * ld + c0


if __name__ == "__main__":
    calib = fit_calibration()
    save_calibration(calib)
    print(f"Calibrated on {calib.n_obs} rows for route {calib.route}")
    print(f"WTP ~ Lognormal(mu={calib.wtp_mu:.3f}, sigma={calib.wtp_sigma:.3f}) "
          f"-> median ${calib.wtp_median:.0f}, mean ${calib.wtp_mean:.0f}")
    print(f"Capacity C={calib.capacity}, horizon={calib.horizon_days} days, "
          f"arrival_rate_base={calib.arrival_rate_base:.3f} customers/day")
    print(f"Saved to {CALIBRATION_OUT}")
