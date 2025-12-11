# 🦟 Dengue Cases Forecasting: A Robust Time-Series Approach

This project focuses on building a complete, scientifically robust end-to-end data science pipeline to forecast dengue cases using historical epidemiological and climate data.

The project's foundation is the publicly available dataset sourced from the final thesis (TCC) of a group of students in São Paulo, Brazil. The analysis was refactored to prioritize data integrity and real-world predictive validity.

📊 This project was developed as a personal initiative to apply the knowledge acquired in the IBM Professional Data Scientist Certificate program, with a focus on advanced time-series methodologies.

---

## 📁 Dataset & Original Project Context

**Source:** Harvard Dataverse  
**Original Project (TCC):** Students from São Paulo create algorithm to predict dengue cases

The dataset contains time-series data on:

- Monthly reported dengue cases (`qntd_casos`)
- Climate variables (temperature, precipitation, wind speed)

---

## 📌 Project Structure

```
aedes_analysis_project/
│
├── data/                 # Raw and processed datasets
├── notebooks/            # Jupyter Notebooks
│   └── aedes_analysis.ipynb   # Main pipeline
├── models/               # Saved plots and model outputs
├── scripts/              # Cleaning and preprocessing scripts
├── README.md             # Project overview
└── requirements.txt      # Python dependencies
```

---

## 🚀 Quick Start: Setup and Execution

Follow these steps to clone the repository, install dependencies, and run the analysis pipeline.

### 1. Clone the Repository

```bash
git clone https://github.com/LucasGalvano/aedes_analysis_project.git
cd aedes_analysis_project/
```

### 2. Install Dependencies

Ensure Python 3.8+ is installed:

> **Note:** During development, Python **3.12.0** was used without issues.

```bash
pip install -r requirements.txt
```

### 3. Run the Analysis

The complete forecasting pipeline (data cleaning, feature engineering, training, and evaluation) is executed through the scripts located in the `scripts/` folder:

- `cleaning_data.py`
- `eda.py`
- `train_dengue_model.py`

To run the full workflow:

```bash
python scripts/cleaning_data.py
python scripts/eda.py
python scripts/train_dengue_model.py
```

Model outputs, plots, and diagnostics are generated inside the `models/` directory.

---

## 🔍 Methodology: Focus on Integrity

A rigorous time-series machine learning workflow was followed, emphasizing scientific soundness.

### 1. Data Cleaning & Leakage Prevention

- **Critical:** Removed 9 symptom-related and test-result features  
  (e.g., `qntd_febre`, `qntd_resultado_ns1`) to eliminate data leakage and ensure the model uses only historically available predictive signals.
- **Robust Imputation:** Time-aware imputation (seasonal medians, FFILL, BFILL) was applied to avoid injecting future information.

### 2. Exploratory Data Analysis (EDA)

- Detected strong seasonal behavior in dengue incidence.
- Confirmed that climate variables affect cases with biological time delays.

### 3. Feature Engineering

- Lag features (1–12 months) generated for cases, precipitation, temperature, wind speed.
- Month and cyclical temporal encodings (sin/cos) created for seasonality capture.

### 4. Machine Learning & Temporal Validation

- **Validation:** Strict `TimeSeriesSplit` ensuring models are always trained on past data and evaluated on unseen future data.
- **Model:** Optimized Ensemble Regressor (XGBoost + Random Forest + GBM) to maximize predictive stability and accuracy.

---

## 🛠️ Technologies Used

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-1D96D2?style=for-the-badge&logo=xgboost&logoColor=white">
  <img src="https://img.shields.io/badge/LightGBM-005C98?style=for-the-badge&logo=lightgbm&logoColor=white">
</p>

---

## 📈 Final Results (Optimized Ensemble Model)

By removing leakage and relying exclusively on lagged predictive features, the Ensemble Model reached strong and reliable performance on the final test set (latest 20% of data).

| Metric               | Value  | Interpretation |
|---------------------|--------|----------------|
| **R² Score**        | 0.753  | 75.3% variance explained |
| **RMSE**            | 7.29   | Avg. magnitude of error |
| **MAE**             | 4.20   | Avg. absolute error |
| **Outbreak F1-Score** | 0.808 | High accuracy in detecting outbreak periods |

**Diagnostic Analysis:**  
Residuals are randomly distributed and predictions follow the actual series closely, confirming model validity.

<div align="center">
  <img src="./models/ensemble_diagnostic_analysis.jpg" alt="Optimized Ensemble Model Diagnostic Analysis" width="700">
</div>

---

## 📬 Contact

**Author:** Lucas Galvano de Paula  
**Email:** lucasgalvano.lgp@gmail.com