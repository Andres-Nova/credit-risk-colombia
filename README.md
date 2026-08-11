# Dashboard de Riesgo Crediticio — Colombia

Análisis de riesgo crediticio con contexto colombiano: pipeline de datos,
comparación de 4 modelos ML, explicabilidad con SHAP y dashboard interactivo.

## 🚀 Demo

- **Dashboard interactivo:** *(enlace tras deploy en Streamlit Cloud)*
- **Reporte completo:** *(enlace tras activar GitHub Pages en `/report`)*

## 📊 Resultados del modelo

> *Completa esta tabla después de ejecutar el notebook 03.*

| Modelo | AUC-ROC | KS | Gini | PR-AUC |
|---|---|---|---|---|
| LightGBM | — | — | — | — |
| XGBoost | — | — | — | — |
| Random Forest | — | — | — | — |
| Regresión Logística | — | — | — | — |

## 📁 Origen de los datos

Los datos base provienen de **Give Me Some Credit** (Kaggle,
[kaggle.com/c/GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit)).
Las variables han sido renombradas al español y enriquecidas con features de
contexto colombiano: estrato socioeconómico simulado, DTF histórica del
Banco de la República, capacidad de pago e historial de mora relativo.
El target (`default`) es observado, no generado artificialmente.

Ver la metodología completa en el [reporte estático](#).

## ⚙️ Setup

```bash
# 1. Clonar e instalar
git clone https://github.com/Andres-Nova/credit-risk-colombia
cd credit-risk-colombia
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Descargar datos desde Kaggle (requiere ~/.kaggle/kaggle.json)
kaggle competitions download -c GiveMeSomeCredit
unzip GiveMeSomeCredit.zip -d data/raw/

# 3. Ejecutar notebooks en orden (ver sección abajo)
```

## 📓 Ejecutar notebooks

```bash
# Ejecutar en orden — cada uno depende del anterior
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb --output notebooks/02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_model_comparison.ipynb --output notebooks/03_model_comparison.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_shap_analysis.ipynb --output notebooks/04_shap_analysis.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_scorecard.ipynb --output notebooks/05_scorecard.ipynb

# Limpiar outputs antes del commit
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

El notebook 03 genera `models/best_model.pkl` y `data/processed/dataset_features.parquet`.

## 🌐 Dashboard local

```bash
cd dashboard && streamlit run app.py
```

## 🧪 Tests

```bash
pytest tests/ -v
```

## 🚀 Deploy

### Streamlit Cloud

1. Ir a [share.streamlit.io](https://share.streamlit.io)
2. New app → `Andres-Nova/credit-risk-colombia` → rama `main` → `dashboard/app.py`
3. Copiar la URL generada y actualizar este README

### GitHub Pages (reporte estático)

1. `Settings → Pages → Source: Deploy from branch → main → /report`
2. Para actualizar el reporte: `Actions → generar-reporte → Run workflow`
3. Requiere secrets en el repo: `KAGGLE_USERNAME` y `KAGGLE_KEY`

## 📂 Estructura

```
credit-risk-colombia/
├── data/
│   ├── raw/              # cs-training.csv (NO commiteado)
│   ├── processed/        # dataset con features colombianas (NO commiteado)
│   └── external/         # dtf_historico.csv, estratos_mapping.csv
├── notebooks/            # 01_eda → 05_scorecard
├── src/
│   ├── data/preprocess.py        # limpieza + renombramiento + split
│   ├── features/build_features.py # 6 features colombianas
│   ├── models/train.py            # 4 modelos + pipelines
│   ├── models/evaluate.py         # KS, AUC, Gini, SHAP, scorecard
│   └── report/generate_report.py  # genera report/index.html
├── dashboard/            # Streamlit — 3 páginas
├── report/               # GitHub Pages (index.html generado)
├── models/               # best_model.pkl (commiteado tras training)
├── tests/                # 8 tests unitarios
└── .github/workflows/    # test.yml + report.yml
```
