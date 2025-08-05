# 🦟 Dengue Cases Forecasting - Data Science Project

This project aims to build a complete end-to-end data science pipeline to forecast dengue cases using historical epidemiological and climate data. It was inspired by the final thesis of a group of students from São Paulo, Brazil, whose dataset served as the foundation for this analysis.

> 📊 This project was developed as a personal initiative to put into practice the knowledge acquired in the **IBM Professional Data Scientist Certificate** program.

---

## 📁 Dataset

- **Source:** [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NN7EOY)
- **Original Project:** [Students from São Paulo create algorithm to predict dengue cases](https://www.metropoles.com/sao-paulo/estudantes-de-sp-criam-algoritmo-capaz-de-prever-casos-de-dengue)

The dataset contains time-series data on:
- Monthly reported dengue cases
- Climate variables (temperature, precipitation, wind speed)

---

## 📌 Project Structure
```
aedes_analysis_project/
│
├── data/ # Raw and processed datasets
├── notebooks/ # Jupyter Notebook ([main pipeline](./notebooks/aedes_analysis.ipynb))
├── models/ # Saved plots and model outputs
├── scripts/ # Scripts for cleaning and preprocessing
├── README.md # Project overview
└── requirements.txt # Python dependencies
```

---

## 🔍 Methodology
Our approach followed a standard machine learning workflow, divided into the following key stages:

### 1. Data Cleaning & Wrangling
- Removed null rows, fixed units, standardized columns
- Filtered dataset to relevant time period
- Transformed temperature and precipitation to proper scales

### 2. Exploratory Data Analysis (EDA)
- Identified seasonal trends and correlations
- Visualized dengue peaks and lagged impact of climate

### 3. Feature Engineering
- Created **lag features** (3 and 4 months) for dengue cases and climate variables

### 4. Machine Learning
- **Baseline model**: Random Forest Regressor
- **Evaluation**: MSE and R² Score
- **Hyperparameter tuning**: RandomizedSearchCV with TimeSeriesSplit

---

## 🛠️ Technologies Used

This project leverages the following tools and libraries:

- <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
- <img src="https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Notebook">
- <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
- <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
- <img src="https://img.shields.io/badge/Matplotlib-E34D26?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
- <img src="https://img.shields.io/badge/Seaborn-4C766A?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn">
- <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">


---

## 📈 Results

| Model        | MSE           | R² Score |
|--------------|---------------|----------|
| Baseline RF  | 6,212,140.50  | 0.14     |
| Optimized RF | 6,202,936.46  | 0.14     |
| CV R² (Best) | -             | **0.30** |

_The "CV R² (Best)" represents the highest R² score achieved during cross-validation, indicating the model's potential._

Despite a modest test R² score, visual inspection confirms accurate timing of dengue outbreaks, especially the 2024 peak. The cross-validation R² suggests the model generalizes better than test scores reflect.

<div align="center">
  <img src="https://raw.githubusercontent.com/LucasGalvano/sao-paulo-dengue-analysis-and-forecasting/main/models/optimized_dengue_forecast_complete.png" alt="Optimized Model's Dengue Forecast" width="600">
</div>

---

## 🚀 Future Improvements

- Implement **XGBoost** for better performance
- Add new features: population density, public health interventions, etc.

---

## 💻 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
``` 

---

## 📬 Contact
Author: Lucas Galvano de Paula

Email: lucasgalvano.lgp@gmail.com

