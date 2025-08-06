import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
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
    df = pd.read_csv(file_path, sep=';', parse_dates=['dt_notificacao'])
    print("Data loaded successfully.")
    return df

def add_features_and_prepare_data(df):
    df.sort_values(by=['cd_municipio', 'dt_notificacao'], inplace=True)

    # Lag features
    lags = [1, 2, 3, 4, 6, 12]
    for lag in lags:
        df[f'qntd_casos_lag{lag}'] = df.groupby('cd_municipio')['qntd_casos'].shift(lag)
        df[f'precipitacao_lag{lag}'] = df.groupby('cd_municipio')['precipitacao_total_mensal'].shift(lag)
        df[f'temp_media_lag{lag}'] = df.groupby('cd_municipio')['temp_media_mensal'].shift(lag)
        df[f'vento_media_lag{lag}'] = df.groupby('cd_municipio')['vento_vlc_media_mensal'].shift(lag)

    # Rolling means
    windows = [3, 6]
    for w in windows:
        df[f'qntd_casos_rm{w}'] = df.groupby('cd_municipio')['qntd_casos'].shift(1).rolling(w).mean()

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df.dropna(inplace=True)

    features = [col for col in df.columns if col not in ['qntd_casos', 'dt_notificacao', 'cd_municipio']]
    X = df[features]
    y = df['qntd_casos']
    
    test_size = 0.2
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    dates_test = df['dt_notificacao'].iloc[split_index:]

    return X_train, X_test, y_train, y_test, dates_test

def evaluate_model(model, X_test, y_test, dates, label):
    y_pred = np.maximum(model.predict(X_test), 0)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{label} -> MSE: {mse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f}")
    return {"MSE": mse, "MAE": mae, "R²": r2, "Dates": dates, "Actual": y_test, "Pred": y_pred}


def plot_comparison(results, dates_test, y_test, models_dir):
    # Plot 1: All models in one plot
    plt.figure(figsize=(16, 8))
    plt.plot(dates_test, y_test, label="Actual", color="black", linewidth=2)
    for label, res in results.items():
        plt.plot(res["Dates"], res["Pred"], label=label, linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Dengue Cases")
    plt.title("Model Forecast Comparison - All Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(models_dir / "model_comparison_all.png", dpi=300)
    plt.close()
    
    # Plot 2: Grid of models
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), sharex=True, sharey=True)
    fig.suptitle('Individual Model Forecast Comparison', fontsize=16)
    axes = axes.flatten()
    
    colors = ['blue', 'orange', 'green', 'red']
    
    for i, (label, res) in enumerate(results.items()):
        ax = axes[i]
        ax.plot(dates_test, y_test, label="Actual", color="black", linewidth=2)
        ax.plot(res["Dates"], res["Pred"], label=label, color=colors[i], linestyle='--')
        ax.set_title(label)
        ax.legend()
        ax.grid(True)
    
    for ax in axes.flat:
        ax.set(xlabel='Date', ylabel='Dengue Cases')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(models_dir / "model_comparison_grid.png", dpi=300)
    plt.show()

# ------------------- MAIN COMPARISON LOGIC -------------------

def run_model_comparison():
    models_dir, data_dir = ensure_dirs()
    df = load_cleaned_data()
    X_train, X_test, y_train, y_test, dates_test = add_features_and_prepare_data(df)
    results = {}
    
    # 1. RF Baseline
    rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    results["RF Baseline"] = evaluate_model(rf_base, X_test, y_test, dates_test, "RF Baseline")

    # 2. RF Optimized
    rf_params = {
        'n_estimators': [200, 500],
        'max_depth': [10, None], 'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2], 'bootstrap': [True],
        'max_features': ['sqrt', None]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    rf_opt_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=rf_params, n_iter=10, cv=tscv, random_state=42,
        n_jobs=-1, scoring='r2'
    )
    rf_opt_search.fit(X_train, y_train)
    results["RF Optimized"] = evaluate_model(rf_opt_search.best_estimator_, X_test, y_test, dates_test, "RF Optimized")

    # 3. XGB Baseline
    xgb_base = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.01,
        random_state=42, n_jobs=-1
    )
    xgb_base.fit(X_train, y_train)
    results["XGB Baseline"] = evaluate_model(xgb_base, X_test, y_test, dates_test, "XGB Baseline")

    # 4. XGB Optimized
    xgb_params = {
        'n_estimators': [500, 1000], 'learning_rate': [0.01, 0.02],
        'max_depth': [5, 7], 'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0], 'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1], 'reg_lambda': [1, 2],
        'min_child_weight': [1, 3]
    }
    xgb_opt_search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
        param_distributions=xgb_params, n_iter=10, cv=tscv, random_state=42,
        n_jobs=-1, scoring='r2'
    )
    xgb_opt_search.fit(X_train, y_train)
    results["XGB Optimized"] = evaluate_model(xgb_opt_search.best_estimator_, X_test, y_test, dates_test, "XGB Optimized")

    # Plot and Save Results
    plot_comparison(results, dates_test, y_test, models_dir)
    
    metrics_df = pd.DataFrame({label: {k: v for k, v in res.items() if k in ["MSE", "MAE", "R²"]} for label, res in results.items()}).T
    metrics_df.to_csv(data_dir / "model_comparison_metrics.csv", index=True)
    print("Metrics saved to models/model_comparison_metrics.csv")


# ------------------- MAIN -------------------

if __name__ == "__main__":
    run_model_comparison()