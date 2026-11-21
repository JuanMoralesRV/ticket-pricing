"""Extrae ATL–BOS del prefijo del ZIP y ajusta el indicador temporal de demanda."""
import argparse
import csv
import hashlib
import json
import struct
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE = "https://www.kaggle.com/datasets/dilwong/flightprices"
COLUMNS = ["searchDate", "flightDate", "segmentsAirlineCode", "segmentsDepartureTimeEpochSeconds",
           "fareBasisCode", "totalFare", "seatsRemaining"]
SAMPLE = DATA / "ATL_BOS.csv"


def extract(path, sample_path):
    digest = hashlib.sha256()
    dates = {}
    scanned = 0
    with open(path, "rb") as raw, sample_path.open("w", encoding="utf-8", newline="") as out:
        prefix = raw.read(30)
        if prefix[:4] != b"PK\x03\x04" or struct.unpack_from("<H", prefix, 8)[0] != 8:
            raise ValueError("Se esperaba el ZIP oficial con compresión DEFLATE.")
        n, extra = struct.unpack_from("<HH", prefix, 26)
        header = raw.read(n + extra)
        if header[:n] != b"itineraries.csv":
            raise ValueError("El primer miembro debe ser itineraries.csv")
        digest.update(prefix + header)
        decoder = zlib.decompressobj(-15)
        tail = b""
        fields = None
        writer = csv.writer(out)
        writer.writerow(COLUMNS)
        while block := raw.read(262144):
            digest.update(block)
            lines = (tail + decoder.decompress(block)).split(b"\n")
            tail = lines.pop()
            if fields is None:
                fields = next(csv.reader([lines.pop(0).decode().strip()]))
            scanned += len(lines)
            for line in lines:
                if b",ATL,BOS," not in line:
                    continue
                row = dict(zip(fields, next(csv.reader([line.decode().strip()]))))
                if (row["startingAirport"], row["destinationAirport"], row["isNonStop"],
                    row["isBasicEconomy"], row["segmentsCabinCode"]) != ("ATL", "BOS", "True", "False", "coach"):
                    continue
                writer.writerow([row[k] for k in COLUMNS])
                dates[row["searchDate"]] = dates.get(row["searchDate"], 0) + 1
    return dict(source=SOURCE, bytes=path.stat().st_size, sha256_prefix=digest.hexdigest(),
                complete_rows_scanned=scanned, route_rows_by_search_date=dates)


def calibrate(sample_path=SAMPLE):
    df = pd.read_csv(sample_path, parse_dates=["searchDate", "flightDate"])
    df["days"] = (df.flightDate - df.searchDate).dt.days
    df = df.query("1 <= days <= 60 and totalFare > 0 and seatsRemaining >= 0").copy()
    key = ["flightDate", "segmentsAirlineCode", "segmentsDepartureTimeEpochSeconds", "fareBasisCode"]
    visits = (df.groupby(key + ["searchDate"], as_index=False)
              .agg(seats=("seatsRemaining", "median"), fare=("totalFare", "median"), days=("days", "first"))
              .sort_values(key + ["searchDate"]))
    g = visits.groupby(key)
    visits["gap"] = g.searchDate.diff().dt.days
    visits["drop"] = (-g.seats.diff()).clip(lower=0)
    visits["prior_fare"] = g.fare.shift()
    visits["prior_seats"] = g.seats.shift()
    pairs = visits.loc[visits.gap == 1].copy()
    if len(pairs) < 100 or pairs["drop"].sum() <= 0:
        raise ValueError("Se necesitan más observaciones repetidas para estimar la curva.")
    # Poisson de cuasi-verosimilitud: la respuesta aproxima reservas netas visibles.
    # Incluir los ceros evita condicionar en un descenso. El signo de la pendiente es libre.
    price_ref = float(pairs.prior_fare.median())
    x = np.column_stack([np.ones(len(pairs)), (60 - pairs.days) / 59, np.log(pairs.prior_fare / price_ref)])
    y = pairs["drop"].to_numpy()
    def loss(b):
        eta = x @ b
        return np.exp(eta).sum() - y @ eta, x.T @ (np.exp(eta) - y)
    opt = minimize(loss, (0, 0, 0), jac=True, method="L-BFGS-B", bounds=[(-10, 6), (-6, 6), (-6, 6)])
    if not opt.success:
        raise RuntimeError(opt.message)
    beta = float(opt.x[1])
    # Incertidumbre agrupada por salida física: permite dependencia entre sus tarifas y fechas.
    groups_id = pairs.groupby(key[:3]).ngroup().to_numpy()
    score = x * (y-np.exp(x @ opt.x))[:,None]
    grouped = np.zeros((groups_id.max()+1, x.shape[1]))
    np.add.at(grouped, groups_id, score)
    bread = np.linalg.inv(x.T @ (np.exp(x @ opt.x)[:,None]*x))
    clusters, n = len(grouped), len(y)
    corr = clusters/(clusters-1)*(n-1)/(n-x.shape[1])
    se = np.sqrt((corr*bread @ (grouped.T @ grouped) @ bread)[1,1])
    ratio_ci = np.exp(beta + np.array([-1,1])*1.96*se).tolist()
    bins = [0, 10, 20, 30, 40, 50, 60]
    pairs["bucket"] = pd.cut(pairs.days, bins, labels=False)
    groups = pairs.groupby("bucket").agg(n=("drop","size"), days=("days","mean"), drop=("drop","mean"),
                                         fare=("prior_fare","mean"), seats=("prior_seats","mean"))
    groups["fit"] = np.exp(opt.x[0] + beta * (60-groups.days)/59)
    cfg = dict(n_rows=len(df), n_pairs=len(pairs), n_offers=int(pairs.groupby(key).ngroups),
               search_start=str(df.searchDate.min().date()), search_end=str(df.searchDate.max().date()),
               first_departure=str(df.flightDate.min().date()),last_departure=str(df.flightDate.max().date()),
               capped=float((pairs.prior_seats>=9).mean()), max_visible=int(df.seatsRemaining.max()),
               positive=float((y>0).mean()), fare_ref=price_ref, intercept=float(opt.x[0]), beta=beta,
               fare_coef=float(opt.x[2]), end_start_ratio=float(np.exp(beta)), ratio_ci=ratio_ci, clusters=int(clusters),
               source=SOURCE, sample_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
               buckets=groups.reset_index().to_dict(orient="records"))
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefijo", type=Path, help="Primeros 128 MiB del ZIP de Kaggle")
    args = parser.parse_args()
    # Valida la extracción y el ajuste antes de reemplazar el extracto incluido.
    with TemporaryDirectory(dir=DATA) as temp:
        pending = Path(temp)
        if args.prefijo:
            sample = pending / SAMPLE.name
            meta = extract(args.prefijo, sample)
            (pending / "procedencia.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        else:
            sample = SAMPLE
        cfg = calibrate(sample)
        calibration = pending / "calibracion.json"
        calibration.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.prefijo:
            sample.replace(SAMPLE)
            (pending / "procedencia.json").replace(DATA / "procedencia.json")
        calibration.replace(DATA / "calibracion.json")
        print({k: v for k, v in cfg.items() if k != "buckets"})
