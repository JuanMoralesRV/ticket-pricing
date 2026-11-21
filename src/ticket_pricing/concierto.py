"""Simulación de precios de concierto. Unidades en días, boletos y USD."""
from dataclasses import dataclass, replace
from pathlib import Path
import json

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
CAL = json.loads((ROOT / "data/calibracion.json").read_text())
NAMES = ("Precio fijo", "Regla de ventas", "Planificación diaria")
TUNE = (10731, 120)
SELECT = (34109, 180)
FINAL = (99173, 600)


@dataclass(frozen=True)
class Evento:
    capacidad: int = 2000
    dias: int = 60
    minimo: int = 40
    maximo: int = 200
    paso: int = 10
    demanda: float = 2.2
    mediana: float = 105.0
    sigma: float = 0.42
    referencia: float = 100.0
    theta: float = 0.75
    eta: float = 0.8
    gamma: float = 0.30
    kappa: float = 1.35

    def __post_init__(self):
        if not (self.capacidad > 0 and self.dias >= 2 and 0 < self.minimo < self.maximo and self.paso > 0
                and (self.maximo-self.minimo) % self.paso == 0 and 2 <= (self.maximo-self.minimo)//self.paso+1 <= 41):
            raise ValueError("Los límites deben formar entre 2 y 41 precios, separados por un paso exacto.")
        if not (self.demanda > 0 and self.mediana > 0 and self.sigma > 0 and self.minimo <= self.referencia <= self.maximo
                and 0 <= self.theta <= 1 and self.eta >= 0 and self.gamma >= 0 and self.kappa >= 1):
            raise ValueError("Revisa la demanda, la memoria y el precio inicial de referencia.")

    @property
    def precios(self):
        return np.arange(self.minimo, self.maximo+1, self.paso, dtype=float)

    @property
    def llegadas(self):
        shape = np.exp(CAL["beta"] * np.linspace(0, 1, self.dias))
        return self.capacidad * self.demanda * shape / shape.sum()


def compra(p, r, c):
    barrera = p + c.eta*np.maximum(p-r, 0) - c.gamma*np.maximum(r-p, 0)
    return ndtr((np.log(c.mediana)-np.log(np.maximum(barrera, 1e-12))) / c.sigma)


def rejilla(x, c):
    x = np.asarray(x)
    indice = np.clip(np.floor((x-c.minimo)/c.paso + 0.5), 0, (c.maximo-c.minimo)//c.paso)
    return c.minimo + indice*c.paso


class Compradores:
    """Poisson base + flujo extra independiente. Cada uniforme es el cuantil de un mismo comprador."""
    def __init__(self, c, seed_n):
        seed, self.n = seed_n
        self.dias = []
        for seq, lam in zip(np.random.SeedSequence(seed).spawn(c.dias), c.llegadas):
            rng = np.random.default_rng(seq)
            self.dias.append((self.flujo(rng, lam), self.flujo(rng, lam*(c.kappa-1))))

    def flujo(self, rng, media):
        n = rng.poisson(media, self.n)
        # Los intervalos [i, i+1) separan los compradores de cada escenario.
        u = np.repeat(np.arange(self.n, dtype=float), n) + rng.random(n.sum())
        return np.sort(u), np.r_[0, n.cumsum()][:-1]

    def interesados(self, t, prob, escasez):
        pos = np.arange(self.n) + prob
        (u, start), (v, extra) = self.dias[t]
        return np.searchsorted(u, pos, side="left")-start + escasez*(np.searchsorted(v,pos,side="left")-extra)


def variables_regla(c, t, stock, reciente, acumulado):
    avance = 1 - stock/c.capacidad - acumulado[t]
    ritmo = np.clip(reciente / (c.capacidad/c.dias) - 1, -1, 1)
    return avance, ritmo


def precio_regla(c, t, stock, reciente, params, acumulado):
    ancla, reaccion, final, memoria = params
    avance, ritmo = variables_regla(c, t, stock, reciente, acumulado)
    return rejilla(ancla + reaccion*avance + final*t/(c.dias-1) + memoria*ritmo, c)


class Plan:
    """DLA determinista. Optimiza una secuencia diaria por Bellman con interpolación."""
    def __init__(self, c, n_stock=81, n_ref=31):
        self.c, self.ns, self.nr = c, n_stock, n_ref
        self.grid = c.precios
        self.rates = c.llegadas
        stock = np.linspace(0, c.capacidad, n_stock)[:, None, None]
        ref = np.linspace(c.minimo, c.maximo, n_ref)[None, :, None]
        self.values = np.zeros((c.dias+1, n_stock, n_ref), dtype=float)
        for t in reversed(range(c.dias)):
            self.values[t] = self.opciones(t, stock, ref).max(axis=-1)

    def opciones(self, t, stock, ref):
        c, p = self.c, self.grid
        lam = self.rates[t] * np.where(stock < 0.1*c.capacidad, c.kappa, 1.)
        sales = np.minimum(stock, lam*compra(p, ref, c))
        si = np.clip((stock-sales)*(self.ns-1)/c.capacidad, 0, self.ns-1)
        ri = np.clip((c.theta*ref + (1-c.theta)*p - c.minimo)*(self.nr-1)/(c.maximo-c.minimo), 0, self.nr-1)
        i, j = si.astype(int), ri.astype(int)
        i1, j1 = np.minimum(i+1, self.ns-1), np.minimum(j+1, self.nr-1)
        a, b = si-i, ri-j
        v = self.values[t+1]
        futuro = (1-a)*((1-b)*v[i,j]+b*v[i,j1])+a*((1-b)*v[i1,j]+b*v[i1,j1])
        return p*sales+futuro

    def precio(self, t, stock, ref):
        return self.grid[self.opciones(t, np.asarray(stock)[...,None], np.asarray(ref)[...,None]).argmax(axis=-1)]


def ejecutar(c, buyers, kind, params=None, plan=None, trace=False):
    # Al ajustar, cada fila es una política candidata; cada columna, un escenario común.
    many = kind in ("fixed_batch", "rule_batch")
    rows = len(params) if many else 1
    shape = (rows, buyers.n)
    stock = np.full(shape, c.capacidad, dtype=int)
    ingresos = np.zeros(shape)
    ref = np.full(shape, c.referencia, dtype=float)
    reciente = np.full(shape, c.capacidad/c.dias)
    acumulado = np.r_[0, c.llegadas.cumsum()] / c.llegadas.sum()
    if kind in ("rule_batch", "rule"):
        ajuste = np.asarray(params).reshape(rows, 4).T[:,:,None]
    paths, stocks = [], []
    for t in range(c.dias):
        if kind in ("fixed_batch", "fixed"):
            price = np.broadcast_to(np.asarray(params).reshape(rows,1),shape)
        elif kind in ("rule_batch", "rule"):
            price = precio_regla(c, t, stock, reciente, ajuste, acumulado)
        else:
            price = plan.precio(t, stock, ref)
        if trace:
            paths.append(np.where(stock>0, price, np.nan)[0])
            stocks.append(stock[0].copy())
        sales = np.minimum(stock, buyers.interesados(t, compra(price, ref, c), stock < 0.1*c.capacidad))
        ingresos += price * sales
        stock -= sales
        ref = c.theta*ref + (1-c.theta)*price
        reciente = 0.65*reciente + 0.35*sales
    if many:
        return ingresos.mean(axis=1)
    return dict(ingresos=ingresos[0], lleno=(stock[0]==0), restantes=stock[0],
                precios=np.asarray(paths), stock=np.asarray(stocks))


def ajustar(c):
    compradores = Compradores(c, TUNE)
    grid = c.precios
    scores = ejecutar(c, compradores, "fixed_batch", grid)
    fijo = float(grid[np.argmax(scores)])
    span = c.maximo-c.minimo
    # Contiene siempre la solución fija; las pendientes admiten descuentos y aumentos.
    anchors = np.unique(rejilla(fijo+c.paso*np.array([-2,-1,0,1,2]), c))
    candidates = np.array([(a,b*span,d*span,m*span) for a in anchors
                           for b in (0,0.3,0.9) for d in (-0.20,0,0.20) for m in (0,0.12)])
    scores = ejecutar(c, compradores, "rule_batch", candidates)
    regla = candidates[np.argmax(scores)]
    return fijo, regla


def trio(c, bloque, fijo, regla, plan, trace=False):
    compradores = Compradores(c,bloque)
    return [ejecutar(c,compradores,"fixed",fijo,trace=trace),
            ejecutar(c,compradores,"rule",regla,trace=trace),
            ejecutar(c,compradores,"plan",plan=plan,trace=trace)]


def intervalo(x):
    x=np.asarray(x)
    margen=1.96*x.std(ddof=1)/np.sqrt(len(x))
    return float(x.mean()-margen), float(x.mean()+margen)


def metricas(r):
    x=r["ingresos"]
    return dict(media=float(x.mean()), sd=float(x.std(ddof=1)), p05=float(np.percentile(x,5)),
                p95=float(np.percentile(x,95)), ic=intervalo(x), lleno=float(r["lleno"].mean()),
                vendidos=float(np.mean(r["stock"][0]-r["restantes"])) if r["stock"].size else None)


def estudio(c):
    fijo, regla = ajustar(c)
    plan = Plan(c)
    valid = trio(c, SELECT, fijo, regla, plan)
    mejor = 0
    # Orden de complejidad prefijado. Solo ascender con ventaja pareada de validación.
    for i in (1,2):
        if intervalo(valid[i]["ingresos"]-valid[mejor]["ingresos"])[0] > 0:
            mejor = i
    out = trio(c,FINAL,fijo,regla,plan,True)
    return dict(fijo=fijo,regla=regla,mejor=mejor,plan=plan,resultados=out,
                stats=[metricas(r) for r in out],validacion=[float(r["ingresos"].mean()) for r in valid])


def sensibilidad(c, factor):
    return replace(c,eta=c.eta*factor,gamma=c.gamma*factor)
