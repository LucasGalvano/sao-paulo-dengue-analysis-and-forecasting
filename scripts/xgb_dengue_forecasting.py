import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


# ------------------- UTILITIES -------------------

def ensure_dirs():
    base_path = Path(__file__).parent.parent
    models_dir = base_path / "models"
    data_dir = base_path / "data"
    models_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    return models_dir, data_dir


def load_cleaned_data():
    _, data_dir = ensure_dirs()
    file_path = data_dir / "cleaned_monthly_dengue_cases.csv"
    try:
        df = pd.read_csv(file_path, sep=';', parse_dates=['dt_notificacao'])
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print("File not found.")
        return None


# ------------------- FEATURE ENGINEERING -------------------

def add_lag_features(df, lags=[1, 2, 3, 4, 6, 12]):
    for lag in lags:
        df[f'qntd_casos_lag{lag}'] = df.groupby('cd_municipio')['qntd_casos'].shift(lag)
        df[f'precipitacao_lag{lag}'] = df.groupby('cd_municipio')['precipitacao_total_mensal'].shift(lag)
        df[f'temp_media_lag{lag}'] = df.groupby('cd_municipio')['temp_media_mensal'].shift(lag)
        df[f'vento_media_lag{lag}'] = df.groupby('cd_municipio')['vento_vlc_media_mensal'].shift(lag)
    print("Lag features added.")
    return df


def add_rolling_means(df, windows=[3, 6]):
    for w in windows:
        df[f'qntd_casos_rm{w}'] = df.groupby('cd_municipio')['qntd_casos'].shift(1).rolling(w).mean()
    print("Rolling mean features added.")
    return df


def add_cyclical_month_encoding(df):
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    print("Cyclical month encoding added.")
    return df


def prepare_data(df):
    df.sort_values(by=['cd_municipio', 'dt_notificacao'], inplace=True)
    
    df = add_lag_features(df)
    df = add_rolling_means(df)
    df = add_cyclical_month_encoding(df)
    df.dropna(inplace=True)

    features = [col for col in df.columns if col not in ['qntd_casos', 'dt_notificacao', 'cd_municipio']]
    target = 'qntd_casos'

    X = df[features]
    y = df[target]

    test_size = 0.2
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    dates_test = df['dt_notificacao'].iloc[split_index:]

    return X_train, X_test, y_train, y_test, dates_test


# ------------------- EVALUATION -------------------

def evaluate_and_plot(model, X_test, y_test, dates, title, save_path, csv_path):
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    plt.figure(figsize=(16, 8))
    sns.lineplot(x=dates, y=y_test, label='Actual', marker='o')
    sns.lineplot(x=dates, y=y_pred, label='Predicted', marker='x')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Dengue Cases')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved at {save_path}")
    plt.close()

    # Save predictions as CSV
    pd.DataFrame({
        'date': dates,
        'actual': y_test.values,
        'predicted': y_pred
    }).to_csv(csv_path, index=False, sep=';')
    print(f"Predictions saved at {csv_path}")


# ------------------- TRAINING -------------------

def train_xgboost_baseline():
    models_dir, data_dir = ensure_dirs()
    df = load_cleaned_data()
    if df is None:
        return
    X_train, X_test, y_train, y_test, dates_test = prepare_data(df)

    print(f"\nTraining XGBoost Regressor with {len(X_train)} samples...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.01,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    evaluate_and_plot(
        model, X_test, y_test, dates_test,
        title='Dengue Cases Forecast - XGBoost Baseline (Improved)',
        save_path=models_dir / "baseline_dengue_forecast_xgb_improved.png",
        csv_path=data_dir / "xgb_predictions_baseline.csv"
    )


def optimize_xgboost_hyperparameters():
    models_dir, data_dir = ensure_dirs()
    df = load_cleaned_data()
    if df is None:
        return

    X_train, X_test, y_train, y_test, dates_test = prepare_data(df)

    param_dist = {
        'n_estimators': [500, 1000, 2000],
        'learning_rate': [0.005, 0.01, 0.02],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2],
        'min_child_weight': [1, 3, 5]
    }

    print("\nStarting hyperparameter search with RandomizedSearchCV...")
    tscv = TimeSeriesSplit(n_splits=8)

    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    random_search = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=param_dist,
        n_iter=50,
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        scoring='r2'
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    print("\n--- Optimization Completed ---")
    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best R² in cross-validation: {random_search.best_score_:.2f}")

    evaluate_and_plot(
        best_model,
        X_test,
        y_test,
        dates_test,
        title='Dengue Forecast - Optimized XGBoost (Improved)',
        save_path=models_dir / "optimized_dengue_forecast_xgb_improved.png",
        csv_path=data_dir / "xgb_predictions_optimized.csv"
    )


# ------------------- MAIN -------------------
if __name__ == "__main__":
    train_xgboost_baseline()
    optimize_xgboost_hyperparameters()