"""Genera reports/Informe_Proyecto_B.pdf con los resultados de scripts/evaluar.py."""

import csv
import json
from pathlib import Path

import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "Informe_Proyecto_B.pdf"
out = json.loads((ROOT / "resultados" / "resumen.json").read_text(encoding="utf-8"))
cal = json.loads((ROOT / "data" / "calibracion.json").read_text(encoding="utf-8"))
base = out["base"]
stats = base["stats"]
cfg = base["config"]
winner = base["choice"]

fonts = Path(reportlab.__file__).parent / "fonts"
pdfmetrics.registerFont(TTFont("Aforo", str(fonts / "Vera.ttf")))
pdfmetrics.registerFont(TTFont("Aforo-Bold", str(fonts / "VeraBd.ttf")))
pdfmetrics.registerFontFamily("Aforo", normal="Aforo", bold="Aforo-Bold")

names = ["Precio fijo", "Regla de ventas", "Planificación diaria"]
pals = ["#656571", "#008276", "#624BB8"]
ink, accent, muted, line = "#25233A", "#51419D", "#535367", "#DCD9E5"
story = []


def color(value):
    return colors.HexColor(value)


def num(value):
    return f"{value:,.0f}"


def usd(value):
    return "$" + num(value)


def pc(value):
    return f"{100 * value:.1f}%"


def ci(bounds):
    return " a ".join(usd(v) for v in bounds)


def ps(size=9.5, lead=13.35, tc=ink, font="Aforo"):
    return ParagraphStyle("p", fontName=font, fontSize=size, leading=lead, textColor=color(tc))


def p(text, style=None, gap=5):
    story.extend([Paragraph(text, style or ps()), Spacer(1, gap)])


def h(text):
    p(text, ps(13.2, 16.5, accent, "Aforo-Bold"), 5)


def title(section, heading, subtitle):
    p(section.upper(), ps(9, 13, accent, "Aforo-Bold"), 5)
    p(heading, ps(23, 27, ink, "Aforo-Bold"), 6)
    p(subtitle, ps(10.5, 14.5, muted), 14)


def tab(header, rows, widths):
    table_header = [Paragraph(str(v), ps(8.6, 11, "#FFFFFF", "Aforo-Bold")) for v in header]
    table_rows = [[Paragraph(str(v), ps(8.7, 11.7)) for v in row] for row in rows]
    table = Table([table_header] + table_rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), color("#39324E")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [color("#F1F0F6"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, color(line)),
    ]))
    story.extend([table, Spacer(1, 12)])


def small(text):
    p(text, ps(8.7, 11.7, muted), 8)


def link(url, label):
    return f'<a href="{url}" color="{accent}"><u>{label}</u></a>'


def page():
    story.append(PageBreak())


class Trayectoria(Flowable):
    def wrap(self, available_width, available_height):
        return 506, 158

    def draw(self):
        canvas = self.canv
        with (ROOT / "resultados" / "trayectorias_base.csv").open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        def x_pos(day):
            return 37 + (day - 1) * 448 / (cfg["dias"] - 1)

        def y_pos(price):
            return 29 + (price - cfg["minimo"]) * 98 / (cfg["maximo"] - cfg["minimo"])

        canvas.setFont("Aforo", 8)
        canvas.setFillColor(color(muted))
        canvas.drawString(1, 146, "Precio mediano entre conciertos con boletos disponibles (USD)")
        for price in range(cfg["minimo"], cfg["maximo"] + 1, 40):
            y = y_pos(price)
            canvas.setStrokeColor(color(line))
            canvas.setLineWidth(0.5)
            canvas.line(37, y, 485, y)
            canvas.drawRightString(29, y - 2, str(price))
        for day in [1, 15, 30, 45, cfg["dias"]]:
            canvas.drawCentredString(x_pos(day), 16, str(day))
        for name, tone, dash in zip(names, pals, ([4, 3], [1, 2], [])):
            coords = [(x_pos(int(row["dia"])), y_pos(float(row["mediana"])))
                      for row in rows if row["estrategia"] == name]
            if not coords:
                continue
            path = canvas.beginPath()
            first_x, previous_y = coords[0]
            path.moveTo(first_x, previous_y)
            for x, y in coords[1:]:
                path.lineTo(x, previous_y)
                path.lineTo(x, y)
                previous_y = y
            canvas.setStrokeColor(color(tone))
            canvas.setLineWidth(1.6)
            canvas.setDash(dash)
            canvas.drawPath(path)
        canvas.setDash([])
        for index, name in enumerate(names):
            x = 37 + index * 150
            canvas.setStrokeColor(color(pals[index]))
            canvas.setLineWidth(2)
            canvas.line(x, 5, x + 12, 5)
            canvas.drawString(x + 16, 2, name)


title("Proyecto B   /   Informe para el promotor","Proyecto B · Precios del concierto","Venta de 2,000 boletos durante 60 días. El promotor puede publicar un precio cada mañana y los boletos sobrantes al cierre pierden su valor.")
m = stats[winner]
delta = base["paired"]["2-0"]
tab(["Estrategia recomendada","Precio de apertura","Ingreso esperado","Probabilidad de lleno"],[[names[winner],usd(base["opening"][winner])+" USD",usd(m["media"])+" USD",pc(m["lleno"])]],[153,108,123,122])
h("Decisión para la campaña")
p(f"Abrir a <b>{usd(base['opening'][winner])} por boleto</b> y recalcular el precio con el inventario y el precio recordado al inicio de cada día. En los 600 conciertos de evaluación, la planificación diaria obtiene <b>{usd(delta['media'])} adicionales por concierto</b> frente al mejor precio fijo. El IC pareado del 95% va de {ci(delta['ic'])}.")
story.extend([Trayectoria(),Spacer(1,10)])
h("Qué sacrifica la recomendación")
p(f"El precio fijo agota el recinto en {pc(stats[0]['lleno'])} de los conciertos y la planificación diaria en {pc(m['lleno'])}. La planificación vende en promedio {m['vendidos']:,.1f} boletos. Se puede quedar a unos pocos boletos del lleno y aun así obtener un ingreso mayor.")
h("Cuándo cambiaría la decisión")
p(f"Si el objetivo fuera maximizar la frecuencia de lleno, la regla de ventas obtiene {pc(stats[1]['lleno'])} en este escenario. Si planificar cada día cuesta más de {usd(base['paired']['2-1']['media'])} adicionales por concierto respecto de esa regla, su ventaja media bruta desaparece. Los costos reales de operación requieren información del promotor.")
small("Todos los importes corresponden a USD e ingreso bruto simulado. Los resultados describen el escenario especificado y su incertidumbre Monte Carlo. El concierto es un caso académico.")
page()
title("Evidencia   /   Calibración","De vuelos a llegadas al concierto","La forma temporal proviene de disponibilidad y tarifas reales. El volumen del público y su disposición a pagar se fijan por separado para el concierto.")
h("Muestra y construcción")
p(f"Se leyó un prefijo de 128 MiB del ZIP público de Expedia en Kaggle. Se conservaron las cotizaciones directas ATL–BOS en cabina económica no básica, con 1 a 60 días de anticipación y tarifa positiva. La muestra tiene {num(cal['n_rows'])} filas entre el 16 y el 20 de abril de 2022. El muestreo por prefijo limita su representatividad.")
p(f"Para cada fecha de salida, aerolínea, hora de salida y código de tarifa, se toma la mediana de asientos visibles y tarifa de ese día. Se comparan solo consultas consecutivas separadas por un día. Hay {num(cal['n_pairs'])} pares en {num(cal['n_offers'])} ofertas. Se calcula el descenso positivo de asientos y se conservan los ceros. Los aumentos de inventario se codifican como descenso cero.")
tab(["Días hasta el vuelo","Pares","Asientos previos¹","Tarifa previa (USD)¹","Descenso por día¹"],
    [[f"{int(b['bucket'])*10+1}–{(int(b['bucket'])+1)*10}",int(b["n"]),f"{b['seats']:.2f}",f"{b['fare']:.2f}",f"{b['drop']:.2f}"] for b in cal["buckets"]],[105,55,103,128,115])
small("¹ Medias sobre los pares de cada intervalo. Una cotización es una oferta publicada. Reasignaciones de tarifa, cupos visibles y cambios de disponibilidad pueden generar movimientos sin una venta.")
h("Ajuste e interpretación")
p(f"Se estima una regresión Poisson con enlace logarítmico. Los predictores son cercanía normalizada a la salida, (60 − días restantes)/59, y logaritmo de la tarifa previa dividida entre USD {cal['fare_ref']:.2f}. La pendiente temporal estimada es {cal['beta']:.3f}. A tarifa constante, la razón entre intensidad final e inicial es <b>{cal['end_start_ratio']:.2f}×</b>. El IC robusto del 95%, agrupado por {cal['clusters']} salidas físicas, es {cal['ratio_ci'][0]:.2f}–{cal['ratio_ci'][1]:.2f} e incluye 1.")
p("Para el concierto se normaliza esa curva hasta sumar 2.2 × 2,000 = 4,400 llegadas base en 60 días. Los topes de asientos visibles y la ausencia de registros de compras impiden identificar el número absoluto de interesados a partir de estos datos. La forma se usa como aproximación transferible.")
small(link(cal["source"],"Expedia · Flight Prices en Kaggle")+"  ·  El extracto, la procedencia y los parámetros se incluyen en data/.")
page()
title("Modelo   /   Powell","Cómo se toma la decisión","Se observa el estado de la mañana, se publica un precio, llegan compradores y se actualiza el estado del día siguiente.")
tab(["Elemento","Definición operacional"],[
["Estado conocido","Día t (1 a T), boletos disponibles I, precio de referencia r (USD por boleto) y media móvil m de ventas (boletos por día). Al inicio I=2,000, r=100 y m=C/T."],
["Decisión","Un precio diario p de la misma rejilla para todas las políticas. Caso base de USD 40 a 200, en pasos de USD 10. La taquilla cierra para ventas al agotar los boletos."],
["Información exógena","Llegadas Poisson diarias y disposición a pagar individual lognormal. Mediana USD 105 y desviación de log(WTP) 0.42. Se revela después de publicar el precio."],
["Transición","Una persona compra si WTP ≥ p + eta·max(p−r,0) − gamma·max(r−p,0). Se venden como máximo I boletos. I(t+1)=I−ventas, r(t+1)=theta·r+(1−theta)·p y m(t+1)=0.65m+0.35·ventas."],
["Objetivo","Maximizar la suma esperada de precio × ventas diarias hasta T. Valor residual de inventario cero. También se informa dispersión de ingreso y probabilidad de agotar el recinto."]],[118,388])
p("La tasa base diaria es proporcional a exp(b·(t−1)/(T−1)), con b estimado en la ruta. Si I/C &lt; 0.10 al abrir el día se multiplica la tasa por 1.35. Los supuestos base de memoria son theta=0.75, eta=0.80 y gamma=0.30. El promotor puede modificarlos.")
h("Las tres políticas")
p(f"<b>Precio fijo.</b> Se ensaya cada precio de la rejilla en los mismos 120 conciertos de ajuste y se conserva el de mayor ingreso medio. El óptimo entre los precios probados fue {usd(base['fixed'])}.")
p("<b>Regla de ventas (PFA).</b> Parte de un precio ancla y añade términos de avance sobre la curva esperada de ventas, fracción de días transcurridos y ritmo de ventas reciente. Redondea el resultado a la rejilla. La búsqueda incluye anclas vecinas al mejor precio fijo y combinaciones prefijadas de coeficientes. El fijo también pertenece a esta familia.")
p(f"Coeficientes ajustados, en USD por unidad del indicador: ancla {base['rule'][0]:.0f}, reacción al avance {base['rule'][1]:.0f}, tiempo {base['rule'][2]:.0f} y ritmo reciente {base['rule'][3]:.1f}. El ritmo se escala por C/T y su desviación se limita al intervalo [−1, 1].")
p("<b>Planificación diaria (DLA).</b> Para cada precio candidato, pronostica ventas con la tasa esperada y la probabilidad de compra. Optimiza una secuencia hasta el cierre con inventario limitado y referencia evolutiva. Se guardan valores sobre 81 niveles de inventario y 31 de referencia, se interpola y solo se aplica la primera decisión. Al día siguiente se vuelve a consultar con el estado observado.")
small("El DLA aproxima las ventas de un día por el mínimo entre demanda esperada e inventario. El simulador final usa compradores individuales aleatorios y aplica el límite físico de boletos.")
page()
title("Resultados   /   Evaluación","Qué tan distintos son los ingresos","Las tres políticas se enfrentan a los mismos compradores potenciales en cada escenario final.")
tab(["Estrategia","Media","Desv. estándar","P5 a P95 por concierto","IC 95% de la media","Lleno"],
    [[n,usd(a["media"]),usd(a["sd"]),ci([a["p05"],a["p95"]]),ci(a["ic"]),pc(a["lleno"])] for n,a in zip(names,stats)],[88,60,68,110,126,54])
p("120 escenarios para ajustar el fijo y la PFA. En un bloque nuevo de 180 escenarios, se parte del fijo y se asciende en complejidad únicamente si el límite inferior del IC pareado del 95% es positivo. La tercera muestra de 600 conciertos evalúa la recomendación ya elegida. Las semillas raíz son 10731, 34109 y 99173, respectivamente.")
p("Cada día y escenario tiene un flujo Poisson común de compradores base y otro adicional que se activa con escasez. Cada comprador tiene un cuantil de disposición a pagar compartido entre políticas. Esto conserva los mismos sucesos potenciales aun cuando los inventarios divergen. El IC se calcula como media ± 1.96 × error estándar. Para comparar estrategias se usa la diferencia dentro del mismo concierto.")
h("Sensibilidad a la referencia y al volumen")
cases=[("Sin efecto de referencia",out["escenarios"]["sin_referencia"]),("Reacciones a la mitad",out["escenarios"]["mitad_referencia"]),("Configuración base",base),("Reacciones ×1.5",out["escenarios"]["mayor_referencia"]),("Demanda baja · 1.2× C",out["escenarios"]["demanda_baja"]),("Demanda alta · 3.2× C",out["escenarios"]["demanda_alta"])]
tab(["Escenario","Política elegida","Apertura","Ingreso medio","Lleno"],[[label,names[z['choice']],usd(z['opening'][z['choice']]),usd(z['stats'][z['choice']]['media']),pc(z['stats'][z['choice']]['lleno'])] for label,z in cases],[146,126,65,99,70])
p("En los cuatro niveles de reacción se mantiene el volumen base y se multiplican eta y gamma. Cada fila vuelve a ajustar, seleccionar y evaluar. Con reacciones intensas, el modelo llega a abrir en USD 200. El mecanismo permite que un precio publicado suba la referencia futura incluso si se venden pocos boletos ese día. Conviene validar esta respuesta con observaciones de conciertos antes de aplicarla en taquilla.")
p("En demanda alta se conserva la política de planificación, pero cambia el precio inicial. Una mejora en ingreso medio puede acompañarse de una menor frecuencia de lleno. Para priorizar lleno o ingresos netos debe cambiarse expresamente el criterio de decisión.")
small("La dispersión entre conciertos y la precisión de una media simulada responden preguntas distintas. Los intervalos Monte Carlo no incorporan el error de trasladar evidencia de vuelos a conciertos.")
page()
title("Demostración   /   Guía del equipo","Cómo explicar el dashboard","Recorrido de 15 minutos desde la decisión de apertura hasta el respaldo del modelo, seguido de cinco minutos para preguntas.")
tab(["Tiempo","Pantalla","Qué demostrar y explicar"],[
["0–2 min","Evaluation (CRN)","Desde Calibration, abrir Evaluation (CRN). Presentar al promotor, el aforo y el objetivo. Leer el precio de apertura, el ingreso esperado y la probabilidad de lleno."],
["2–5 min","Evaluation (CRN)","Explicar las tres políticas en lenguaje de negocio. Contrastar ingreso y lleno, abrir Revenue distribution and paired differences e ilustrar la diferencia entre la dispersión P5–P95 y el IC sobre la media."],
["5–8 min","Episode trace","Mostrar las dos gráficas del mismo concierto y abrir Today’s recommended price. En el escenario base, cambiar al día 30, ingresar 1,600 boletos y precio recordado USD 100. La planificación sugiere USD 90. Cambiar únicamente el inventario a 1,100 eleva la decisión a USD 110."],
["8–11 min","Evaluation (CRN)","Abrir Reference-price sensitivity. Comparar el caso sin efecto de referencia y el configurado. Explicar por qué la política elegida puede mantenerse mientras cambian el precio o los ingresos. Mencionar el costo operativo que haría preferible la regla."],
["11–14 min","Calibration","Mostrar la curva de tarifas ATL–BOS y abrir Concert arrival-rate calibration para enseñar la tabla de asientos y la curva de llegadas. Distinguir la forma estimada del volumen y la disposición a pagar asumidos."],
["14–15 min","Evaluation (CRN)","Cerrar con la decisión de apertura y el dato que pedirían al promotor antes de usarla: historial diario de precios, visitas, ventas y disponibilidad del recinto."]],[66,102,338])
h("Cómo operar los controles")
p("La barra lateral configura el concierto completo. Los cambios de aforo, ventana y precios en <b>Concert &amp; demand assumptions</b> recalculan el escenario. En una pantalla estrecha, la flecha superior izquierda abre los controles. El control <b>Today’s recommended price</b> de la pestaña Episode trace solo consulta qué precio publicar en el estado ingresado.")
h("Preguntas que conviene saber responder")
p("<b>¿De dónde salen los compradores?</b> La forma relativa se estima de cotizaciones repetidas de Expedia. Se asume la escala del concierto. <b>¿Por qué no fijar siempre USD 110?</b> Aunque coinciden en apertura, planificar permite responder a inventario y referencia durante los 60 días. <b>¿Es garantía de ingreso?</b> Es el promedio de un simulador con los supuestos indicados.")
small(link("https://iteso.instructure.com/courses/58266/assignments/1276507","ITESO · consigna de Proyecto B")+"  ·  "+link("https://iteso.instructure.com/courses/58266/modules/items/2543626","Material de clase sobre políticas")+"  ·  "+link(cal["source"],"Datos de Expedia"))


def footer(canvas, doc):
    canvas.setFillColor(color("#373051"))
    canvas.rect(0, 782, 612, 10, fill=1, stroke=0)
    canvas.setStrokeColor(color(line))
    canvas.line(51, 39, 561, 39)
    canvas.setFont("Aforo", 8)
    canvas.setFillColor(color(muted))
    canvas.drawString(51, 25, "PRECIOS DE CONCIERTO")
    canvas.drawRightString(561, 25, f"PROYECTO B     ·     {doc.page}")


pdf = SimpleDocTemplate(
    str(OUTPUT), pagesize=letter, leftMargin=47, rightMargin=47, topMargin=42, bottomMargin=46,
    title="Proyecto B · Precios dinámicos del concierto", invariant=1,
)
pdf.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Informe generado en {OUTPUT}")
