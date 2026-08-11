# Dashboard de Riesgo Crediticio — Colombia

Análisis de riesgo crediticio con contexto colombiano: pipeline de datos,
comparación de 4 modelos ML, explicabilidad con SHAP y dashboard interactivo.

## 🚀 Demo

- **Dashboard interactivo:** *(disponible tras deploy en Streamlit Cloud)*
- **Reporte completo:** *(disponible tras activar GitHub Pages)*

## 📊 Resultados del modelo

| Modelo | AUC-ROC | KS | Gini | PR-AUC |
|---|---|---|---|---|
| *Pendiente tras entrenamiento* | | | | |

## 📁 Origen de los datos

Los datos base provienen de **Give Me Some Credit** (Kaggle,
[kaggle.com/c/GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit)).
Las variables han sido renombradas al español y enriquecidas con features de
contexto colombiano: estrato socioeconómico simulado, DTF histórica del
Banco de la República, capacidad de pago e historial de mora relativo.
El target (`default`) es observado — no generado artificialmente.

Ver metodología completa en el reporte estático.

## ⚙️ Setup

```bash
# 1. Clonar e instalar
git clone https://github.com/Andres-Nova/credit-risk-colombia
cd credit-risk-colombia
pip install -r requirements.txt

# 2. Descargar datos desde Kaggle (requiere API key en ~/.kaggle/kaggle.json)
kaggle competitions download -c GiveMeSomeCredit
unzip GiveMeSomeCredit.zip -d data/raw/

# 3. Ejecutar notebooks en orden
jupyter notebook notebooks/
```

## 📂 Estructura

```
credit-risk-colombia/
├── data/
│   ├── raw/              # cs-training.csv (NO commiteado — ver instrucciones)
│   ├── processed/        # dataset limpio + features (NO commiteado)
│   └── external/         # dtf_historico.csv, estratos_mapping.csv
├── notebooks/            # 01_eda → 05_scorecard
├── src/
│   ├── data/preprocess.py
│   ├── features/build_features.py
│   ├── models/train.py + evaluate.py
│   └── report/generate_report.py
├── dashboard/            # Streamlit app (3 páginas)
├── report/               # GitHub Pages (index.html generado)
├── models/               # best_model.pkl + preprocessor.pkl
├── tests/                # pytest
└── .github/workflows/    # CI/CD
```

## 📓 Ejecutar notebooks en orden

```bash
# EDA
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output notebooks/01_eda.ipynb

# Feature engineering
jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb --output notebooks/02_feature_engineering.ipynb

# Entrenar modelos (puede tardar ~10 min)
jupyter nbconvert --to notebook --execute notebooks/03_model_comparison.ipynb --output notebooks/03_model_comparison.ipynb

# SHAP
jupyter nbconvert --to notebook --execute notebooks/04_shap_analysis.ipynb --output notebooks/04_shap_analysis.ipynb

# Scorecard
jupyter nbconvert --to notebook --execute notebooks/05_scorecard.ipynb --output notebooks/05_scorecard.ipynb

# Limpiar outputs antes del commit
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

## 🧪 Tests

```bash
pytest tests/ -v
```
