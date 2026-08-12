# Riesgo Crediticio — Colombia

Construí este proyecto para demostrar un pipeline completo de ciencia de datos aplicado a riesgo crediticio, desde la ingesta y limpieza de datos hasta el despliegue de un dashboard interactivo.

El dataset base es [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, 150k registros con default observado). Lo enriquecí con seis variables de contexto colombiano — estrato socioeconómico simulado, DTF histórica del Banco de la República, capacidad de pago, historial de mora relativo, segmento de edad y score de mora acumulada — para que el modelo tenga semántica local, no solo variables traducidas.

## Demo

- **Dashboard interactivo:** [credit-risk-colombia.streamlit.app](https://credit-risk-colombia.streamlit.app/)
- **Reporte completo:** *(pendiente — GitHub Pages)*

## Resultados

Comparé cuatro modelos controlando el desbalance de clases (93% no-default / 7% default) con estrategias distintas según el algoritmo.

| Modelo | AUC-ROC | KS | Gini | PR-AUC |
|---|---|---|---|---|
| **LightGBM** ⭐ | **0.8604** | **0.5706** | **0.7208** | **0.3956** |
| XGBoost | 0.8580 | 0.5642 | 0.7159 | 0.3910 |
| Random Forest | 0.8577 | 0.5629 | 0.7154 | 0.3812 |
| Regresión Logística | 0.8466 | 0.5458 | 0.6932 | 0.3665 |

LightGBM ganó por margen estrecho. Un KS de 0.57 es sólido para un modelo de scoring — en banca el umbral mínimo aceptable suele estar en 0.30.

La probabilidad de default se transforma a un scorecard 300-850 estilo FICO, con semáforo de riesgo: 🔴 ≤500 / 🟡 501-650 / 🟢 >650.

## Qué modelos comparo y por qué

| Modelo | Rol en la comparación |
|---|---|
| Regresión Logística | Baseline interpretable — base del scorecard tradicional |
| Random Forest | Ensemble robusto sin necesidad de escalar variables |
| XGBoost | Boosting secuencial con `scale_pos_weight` para el desbalance |
| LightGBM | Boosting por hojas — más rápido y mejor AUC en datos tabulares densos |

## Correr localmente

```bash
pip install -r requirements.txt

# Descargar datos (requiere ~/.kaggle/kaggle.json)
kaggle competitions download -c GiveMeSomeCredit
unzip GiveMeSomeCredit.zip -d data/raw/

# Ejecutar notebooks en orden (01 → 03 generan el modelo)
jupyter notebook notebooks/

# Dashboard
streamlit run dashboard/app.py
```

## Tests

```bash
pytest tests/ -v
```
