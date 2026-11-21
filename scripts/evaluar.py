"""Evalúa políticas y exporta las cifras utilizadas por el informe."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ticket_pricing.concierto import Evento, NAMES, ROOT, FINAL, SELECT, TUNE, estudio, intervalo, sensibilidad


def documentar(c):
    s=estudio(c)
    r=s["resultados"]
    gain={}
    for j in range(1,3):
        for i in range(j):
            d=r[j]["ingresos"]-r[i]["ingresos"]
            gain[f"{j}-{i}"]={"media":float(d.mean()),"ic":intervalo(d)}
    return dict(config=asdict(c), fixed=s["fijo"], rule=s["regla"].tolist(), choice=int(s["mejor"]),
                opening=[float(a["precios"][0,0]) for a in r], stats=s["stats"], paired=gain,
                validation=s["validacion"]), s


def main():
    c=Evento()
    base,s=documentar(c)
    results=ROOT/"resultados"
    results.mkdir(exist_ok=True)
    rows=[]
    paths=[]
    for name,r in zip(NAMES,s["resultados"]):
        rows.extend(dict(escenario=j+1,estrategia=name,ingreso_usd=x,boletos_sobrantes=int(y))
                    for j,(x,y) in enumerate(zip(r["ingresos"],r["restantes"])))
        for t,(price,stock) in enumerate(zip(r["precios"],r["stock"])):
            a=price[np.isfinite(price)]
            if len(a):
                paths.append(dict(dia=t+1,estrategia=name,activos=len(a),p10=np.percentile(a,10),
                                  mediana=np.median(a),p90=np.percentile(a,90),inventario_mediano=np.median(stock)))
    pd.DataFrame(rows).to_csv(results/"conciertos_base.csv",index=False)
    pd.DataFrame(paths).to_csv(results/"trayectorias_base.csv",index=False)
    variants={"sin_referencia":sensibilidad(c,0),"mitad_referencia":sensibilidad(c,.5),
              "mayor_referencia":sensibilidad(c,1.5),"demanda_baja":replace(c,demanda=1.2),
              "demanda_alta":replace(c,demanda=3.2)}
    output=dict(tune=TUNE,selection=SELECT,final=FINAL,base=base,
                escenarios={key:documentar(event)[0] for key,event in variants.items()})
    (results/"resumen.json").write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
    print(f"Escenario base: {NAMES[base['choice']]}; {FINAL[1]} conciertos por política.")

if __name__=="__main__": main()
