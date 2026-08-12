# Dashboard de Riesgo Crediticio — Colombia

Comparación de 4 modelos ML para predecir default crediticio, con contexto colombiano (estrato socioeconómico, DTF, capacidad de pago) y scorecard 300-850 estilo FICO.

## Demo

- **Dashboard interactivo:** *(pendiente — Streamlit Cloud)*
- **Reporte completo:** *(pendiente — GitHub Pages)*

## Resultados

| Modelo | AUC-ROC | KS | Gini | PR-AUC |
|---|---|---|---|---|
| **LightGBM** ⭐ | **0.8604** | **0.5706** | **0.7208** | **0.3956** |
| XGBoost | 0.8580 | 0.5642 | 0.7159 | 0.3910 |
| Random Forest | 0.8577 | 0.5629 | 0.7154 | 0.3812 |
| Regresión Logística | 0.8466 | 0.5458 | 0.6932 | 0.3665 |

## Datos

Base: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, 150k registros, target observado).
Variables renombradas al español y enriquecidas con 6 features de contexto colombiano.

## Correr localmente

```bash
pip install -r requirements.txt
kaggle competitions download -c GiveMeSomeCredit && unzip GiveMeSomeCredit.zip -d data/raw/
jupyter notebook notebooks/   # ejecutar 01 → 03 en orden
streamlit run dashboard/app.py
```

## Tests

```bash
pytest tests/ -v
```
