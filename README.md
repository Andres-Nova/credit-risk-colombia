# 💳 Riesgo Crediticio — Colombia

[![Demo LIVE](https://img.shields.io/badge/Demo-LIVE-brightgreen?style=flat-square&logo=streamlit)](https://credit-risk-colombia.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-AUC%200.86-lightgrey?style=flat-square)](https://lightgbm.readthedocs.io)
[![Tests](https://img.shields.io/badge/Tests-8%20passing-brightgreen?style=flat-square)](tests/)

Construí este proyecto para demostrar un pipeline completo de ciencia de datos aplicado a un problema real de banca: predecir qué solicitantes de crédito entrarán en mora, y traducir esa predicción en un número accionable para el oficial de crédito.

El dataset base es [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, 150 mil registros con default observado). Lo enriquecí con seis variables de contexto colombiano — estrato socioeconómico simulado, DTF histórica del Banco de la República, capacidad de pago, historial de mora relativo, segmento de edad y score de mora acumulada — para que el modelo tenga semántica local, no solo variables traducidas.

---

## Demo

- **Dashboard interactivo:** [credit-risk-colombia.streamlit.app](https://credit-risk-colombia.streamlit.app/)
- **Código fuente:** [github.com/Andres-Nova/credit-risk-colombia](https://github.com/Andres-Nova/credit-risk-colombia)

---

## Resultados

Comparé cuatro modelos controlando el desbalance de clases (93 % no-default / 7 % default). Cada algoritmo maneja el desbalance de forma distinta: `class_weight='balanced'` para los modelos de sklearn, `scale_pos_weight` para XGBoost e `is_unbalance=True` para LightGBM.

| Modelo | AUC-ROC | KS | Gini | PR-AUC |
|---|---|---|---|---|
| **LightGBM** ⭐ | **0.8604** | **0.5706** | **0.7208** | **0.3956** |
| XGBoost | 0.8580 | 0.5642 | 0.7159 | 0.3910 |
| Random Forest | 0.8577 | 0.5629 | 0.7154 | 0.3812 |
| Regresión Logística | 0.8466 | 0.5458 | 0.6932 | 0.3665 |

LightGBM ganó por margen estrecho, lo cual es esperado en datos tabulares densos: el boosting por hojas captura mejor las interacciones no lineales que el boosting por nivel de XGBoost.

---

## Qué mide cada métrica — y por qué importa en crédito

Elegí este conjunto de métricas porque ninguna sola es suficiente en un modelo de scoring. Cada una responde una pregunta diferente del negocio.

### AUC-ROC — ¿Qué tan bien separa el modelo buenos de malos pagadores?

El área bajo la curva ROC mide la probabilidad de que el modelo asigne una probabilidad de default más alta a un cliente que realmente entra en mora que a uno que no. Un AUC de 0.86 significa que el modelo acierta ese ranking el 86 % de las veces.

- **AUC = 0.5** → el modelo no discrimina mejor que el azar
- **AUC > 0.8** → umbral mínimo aceptable en muchos comités de riesgo
- **AUC > 0.85** → considerado bueno para datos de crédito masivo
- **AUC = 1.0** → separación perfecta (señal de data leakage)

Es la métrica más usada para comparar modelos, pero no dice nada sobre qué umbral de corte usar en producción.

### KS (Kolmogorov-Smirnov) — ¿En qué punto el modelo separa mejor la cartera?

El estadístico KS mide la máxima separación entre la función de distribución acumulada de buenos pagadores y la de malos pagadores. En scoring crediticio es la métrica operativa por excelencia: identifica el umbral de score en el que la diferencia entre ambas poblaciones es máxima, que es exactamente donde el banco debería poner el corte de aprobación.

- **KS < 0.20** → modelo débil, poco útil en producción
- **KS 0.20-0.40** → aceptable para carteras de bajo riesgo
- **KS 0.40-0.60** → bueno; el modelo discrimina bien
- **KS > 0.60** → excelente

Con KS = 0.57, el modelo está en la banda "buena" y es viable para usarse en una política de aprobación real.

### Gini — Versión normalizada del AUC para reportes ejecutivos

El coeficiente Gini es simplemente `2 × AUC - 1`. Mide qué tan lejos está el modelo del azar (Gini = 0) versus la perfección (Gini = 1). Se usa en reportes de riesgo porque es más intuitivo para audiencias no técnicas: un Gini de 0.72 significa que el modelo explica el 72 % de la separación máxima posible.

Muchos reguladores (Basilea, SFC en Colombia) exigen reportar el Gini como indicador de validación de modelos internos.

### PR-AUC — ¿Qué tan útil es el modelo para encontrar los malos pagadores?

El área bajo la curva Precision-Recall es la métrica más honesta cuando las clases están muy desbalanceadas. Un dataset con 7 % de defaults puede tener un AUC de 0.75 simplemente porque la clase mayoritaria es fácil de predecir. La curva PR no usa los verdaderos negativos, así que no se infla con la clase mayoritaria.

- **PR-AUC basal** (sin modelo) ≈ tasa de default del dataset = 0.067
- **Nuestro PR-AUC = 0.40** → el modelo es ~6× mejor que asignar mora al azar

En detección de fraude o mora esperada, el PR-AUC es el indicador que más le importa al área de recuperación de cartera.

---

## Del modelo al scorecard

La probabilidad de default que entrega el modelo no se usa directamente. La transformé a un scorecard 300-850 estilo FICO usando la transformación log-odds estándar:

```
score = base + factor × log(odds)
factor = PDO / ln(2)         # PDO = 20 puntos que duplican las odds
base   = 600                 # score en odds = 19:1 (95 % buenos / 5 % malos)
```

Esto produce un número continuo con semáforo de riesgo:
- 🔴 ≤ 500 — Alto riesgo
- 🟡 501-650 — Riesgo medio
- 🟢 > 650 — Bajo riesgo

La ventaja de esta transformación es que el score es interpretable sin conocer probabilidades: una diferencia de 20 puntos siempre representa duplicar o dividir a la mitad las odds de mora, independientemente del nivel del score.

---

## Qué modelos comparo y por qué

| Modelo | Rol en la comparación |
|---|---|
| Regresión Logística | Baseline interpretable — base del scorecard tradicional, coeficientes directamente como pesos |
| Random Forest | Ensemble robusto sin necesidad de escalar variables ni manejar outliers |
| XGBoost | Boosting secuencial con `scale_pos_weight` para el desbalance; rápido y regularizable |
| LightGBM | Boosting por hojas (leaf-wise) — más eficiente en datos densos, ganó por AUC y KS |

Elegí estos cuatro porque representan el espectro real de decisiones en un proyecto de scoring: desde el modelo lineal que exige el área de compliance hasta los ensembles que maximizan métricas.

---

## Estructura del proyecto

```
credit-risk-colombia/
├── src/
│   ├── data/preprocess.py          # Limpieza, renombramiento de columnas, split estratificado
│   ├── features/build_features.py  # 6 variables de contexto colombiano
│   ├── models/train.py             # 4 pipelines sklearn (preprocesador + clasificador)
│   └── models/evaluate.py         # KS, AUC, Gini, PR-AUC, scorecard, SHAP
├── dashboard/
│   ├── app.py                      # Entrada multipage Streamlit
│   └── pages/
│       ├── 1_resumen.py            # KPIs, distribución de scores, mora por estrato
│       ├── 2_simulador.py          # Score individual con explicación por variables
│       └── 3_modelo.py             # Curvas ROC, KS plot, aprobación vs mora
├── models/best_model.pkl           # LightGBM serializado (pipeline completo)
├── data/processed/                 # Dataset con features colombianas (.parquet)
├── tests/                          # 8 tests unitarios (pytest)
└── .github/workflows/              # CI en push + reporte manual con secretos Kaggle
```

---

## Correr localmente

```bash
git clone https://github.com/Andres-Nova/credit-risk-colombia.git
cd credit-risk-colombia

pip install -r requirements.txt

# El modelo y los datos ya están en el repo — el dashboard arranca directamente:
streamlit run dashboard/app.py

# Para re-entrenar desde cero (requiere ~/.kaggle/kaggle.json):
kaggle competitions download -c GiveMeSomeCredit
unzip GiveMeSomeCredit.zip -d data/raw/
# Luego ejecutar notebooks 01-03 en orden
```

## Tests

```bash
pytest tests/ -v
# 8 tests: limpieza de datos, ingeniería de features, integridad del split
```

---

## ✍️ Autor

**Andres Nova** — AI Solutions Architect  
[andres-nova.github.io](https://andres-nova.github.io) · [LinkedIn](https://linkedin.com/in/andres-nova-data)
