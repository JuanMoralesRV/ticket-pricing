# Proyecto B · Precios dinámicos para un concierto

**Recomendación para el promotor.** Abrir a **USD 110** y revisar el precio cada mañana con el inventario restante y la referencia del público. En los 600 escenarios comunes de evaluación, la planificación diaria (DLA) genera un ingreso medio de **USD 220,684**, aproximadamente **USD 1,608 más** que el mejor precio fijo. El intervalo del 95% para esa diferencia pareada va de USD 1,484 a USD 1,733. [Informe completo y guía de exposición](Informe_Proyecto_B.pdf)

| Política | Ingreso medio (USD) | P5–P95 por concierto (USD) | IC 95% del ingreso medio (USD) | Probabilidad de lleno |
| --- | ---: | ---: | ---: | ---: |
| Precio fijo | 219,075 | 213,840–220,000 | 218,888–219,262 | 77.3% |
| Regla de ventas (PFA) | 219,298 | 214,998–221,655 | 219,132–219,463 | 78.2% |
| Planificación diaria (DLA) | 220,684 | 216,760–223,941 | 220,512–220,855 | 66.3% |

**Qué cambia la elección.** Si el promotor prioriza agotar los 2,000 boletos, la PFA alcanzó la mayor frecuencia de lleno en esta evaluación. La ventaja media bruta de DLA frente a PFA fue de USD 1,386 por concierto. Un costo adicional de operación superior a esa cifra favorece a la PFA en ingreso neto.

## Modelo y evaluación

Durante los 60 días de venta, el promotor observa el día, inventario, referencia de precio y ventas recientes antes de decidir un precio entre USD 40 y USD 200 en pasos de USD 10. Después llegan compradores con disposición a pagar lognormal, se venden como máximo los boletos disponibles y se actualizan inventario, referencia y ventas recientes. El objetivo es maximizar el ingreso bruto esperado; el inventario sobrante vale cero. El equipo ajustó las políticas con 120 escenarios, eligió la recomendación con otros 180 y la evaluó en 600 escenarios nuevos. [Modelo de Powell, políticas, fórmulas y sensibilidad](Informe_Proyecto_B.pdf) · [Resultados numéricos](../resultados/resumen.json)

## Calibración y alcance

Las tarifas observadas en 2,682 cotizaciones directas ATL–BOS se representan en la primera gráfica. La forma de la curva de llegadas se estima por separado usando los descensos de asientos publicados entre fechas consecutivas. A tarifa constante, el ajuste da una intensidad final de 1.23 veces la inicial, con IC robusto del 95% de 0.76 a 1.98. Las reasignaciones de cupos también afectan los cambios observados. El volumen total de compradores, su disposición a pagar y la respuesta al precio recordado son supuestos del concierto. [Calibración de la ruta](../data/calibracion.json)

La entrega anterior y sus figuras corresponden al caso de un bloque de 10 asientos de avión. [Informe histórico](archivo/informe-vuelo.md)
