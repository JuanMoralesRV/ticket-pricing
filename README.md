# Proyecto B · Precios dinámicos para un concierto

Un promotor tiene 2,000 lugares y 60 días para venderlos. El dashboard de Streamlit compara tres decisiones diarias de precio en USD: mantener un precio fijo, aplicar una regla según las ventas (PFA) o planificar hasta el día del concierto (DLA). El escenario base abre a **USD 110** con planificación diaria. [Informe del proyecto](reports/Informe_Proyecto_B.pdf) · [Resultados del concierto](reports/report.md)

## Abrir el dashboard

Desde la raíz del proyecto, crear un entorno e instalar las dependencias una sola vez.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, activar el entorno con `.venv\Scripts\activate`. Después, iniciar Streamlit con el entorno activo.

```bash
streamlit run dashboard/app.py
```

## Qué mostrar

| Pestaña | Qué permite ver |
| --- | --- |
| **Calibration** | Tarifas observadas de Expedia para Atlanta–Boston. **Concert arrival-rate calibration** muestra el ajuste de llegadas a partir de cambios diarios de disponibilidad. |
| **Evaluation (CRN)** | Ingreso, dispersión, intervalos y probabilidad de lleno de las tres políticas frente a los mismos 600 conciertos simulados. Incluye distribución de ingresos y sensibilidad al precio de referencia. |
| **Episode trace** | Precio e inventario de un concierto, lado a lado. **Today's recommended price** permite consultar el precio que corresponde a un día e inventario concretos. |

Para la demo, conservar los controles iniciales y abrir **Today's recommended price**. En el día 30, con referencia de USD 100 y 1,600 boletos disponibles, la recomendación es **USD 90**. Al cambiar únicamente el inventario a 1,100, sube a **USD 110**.

## Datos y reproducción

La muestra incluida contiene 2,682 cotizaciones reales. La curva de tarifas muestra cómo varía el precio anunciado. Un segundo ajuste estima la forma temporal de las llegadas a partir de 1,149 pares de observaciones consecutivas de disponibilidad. El volumen del concierto y la disposición a pagar son supuestos editables. [Fuente original de Expedia en Kaggle](https://www.kaggle.com/datasets/dilwong/flightprices) · [Procedencia del extracto](data/procedencia.json)

Los datos de la ruta y los resultados base ya están incluidos. Desde la raíz, estos comandos reconstruyen el ajuste, las simulaciones y el PDF, y ejecutan las pruebas.

```bash
python scripts/calibrar.py
python scripts/evaluar.py
python scripts/informe.py
python -m pytest -q
```

El simulador y las políticas del concierto están en [concierto.py](src/ticket_pricing/concierto.py). El informe y la calibración del ejercicio anterior con 10 asientos de avión se conservan como referencia histórica. [Informe del vuelo](reports/archivo/informe-vuelo.md) · [Resultados del vuelo](reports/results.json)
