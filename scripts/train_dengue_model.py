import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    mean_absolute_percentage_error, precision_score, 
    recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings

# --- SETUP ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

LEAKAGE_FEATURES = [
    'qntd_febre', 'qntd_vomito', 'qntd_nausea', 'qntd_sangramento',
    'qntd_resultado_soro', 'qntd_resultado_ns1', 'qntd_resultado_pcr',
    'qntd_hospitalizacao', 'qntd_auctone'
]

TEST_SIZE = 0.2
TOP_K_FEATURES = 25
RANDOM_STATE = 42

def ensure_dirs():
    """Creates necessary directories."""
    base_path = Path.cwd()
    models_dir = base_path / "models"
    data_dir = base_path / "data"
    models_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    return models_dir, data_dir

# --- DATA PROCESSING CLASS ---
class DengueDataProcessor:
    def __init__(self, file_path, target_col='qntd_casos'):
        self.file_path = file_path
        self.target_col = target_col
        # Use SimpleImputer with median strategy for numerical data
        self.imputer = SimpleImputer(strategy='median')
        
    def load_and_clean(self):
        """Loads data and removes leakage features immediately."""
        try:
            df = pd.read_csv(self.file_path, sep='\t', encoding='latin-1', parse_dates=['dt_notificacao'])
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return None

        # 1. Drop rows with missing Target
        df.dropna(subset=[self.target_col], inplace=True)
        
        # 2. Sort by date (Crucial for Time Series)
        df = df.sort_values('dt_notificacao').reset_index(drop=True)
        
        # 3. Remove Leakage Features
        cols_to_drop = [c for c in LEAKAGE_FEATURES if c in df.columns]
        if cols_to_drop:
            print(f"Removing {len(cols_to_drop)} leakage features (symptoms/tests)...")
            df.drop(columns=cols_to_drop, inplace=True)
            
        return df

    def prepare_split(self, df, test_size=0.2):
        """
        Splits data chronologically and handles imputation correctly,
        including dropping constant features to prevent Imputer errors.
        """
        # 1. Define Candidate Features (Exclude ID, Dates, Target)
        exclude_cols = [self.target_col, 'dt_notificacao', 'cd_municipio', 'ano_notificacao', 'month', 'quarter']
        candidate_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        
        # 2. Temporal Split
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        X_train = train_df[candidate_cols]
        X_test = test_df[candidate_cols]
        y_train = train_df[self.target_col]
        y_test = test_df[self.target_col]
        dates_test = test_df['dt_notificacao']
        
        print("\n--- Temporal Split ---")
        print(f"Train Range: {train_df['dt_notificacao'].min().date()} to {train_df['dt_notificacao'].max().date()} ({len(train_df)} samples)")
        print(f"Test Range:  {test_df['dt_notificacao'].min().date()} to {test_df['dt_notificacao'].max().date()} ({len(test_df)} samples)")

        # 3. CRITICAL STEP: Identify and remove constant (zero variance) columns from X_train
        print("Checking and removing constant/all-NaN features...")
        constant_or_nan_cols = [
            col for col in X_train.columns
            if X_train[col].nunique(dropna=False) <= 1 or X_train[col].isnull().all()
        ]
        
        if constant_or_nan_cols:
            print(f" -> Dropping {len(constant_or_nan_cols)} features (constant or all NaN): {constant_or_nan_cols}")
            X_train.drop(columns=constant_or_nan_cols, inplace=True)
            X_test.drop(columns=constant_or_nan_cols, inplace=True)
            
        # 4. Final Feature List
        feature_cols = X_train.columns.tolist()
        
        # 5. Correct Imputation (No Lookahead Bias)
        print(f"Imputing missing values using median (Features used: {len(feature_cols)})...")
        
        # We ensure the columns=feature_cols are used to reconstruct the dataframe
        X_train_imputed = pd.DataFrame(self.imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_test_imputed = pd.DataFrame(self.imputer.transform(X_test), columns=feature_cols, index=X_test.index)
        
        return X_train_imputed, X_test_imputed, y_train, y_test, dates_test, feature_cols

# --- FEATURE SELECTION ---
def hybrid_feature_selection(X_train, y_train, k=TOP_K_FEATURES):
    """
    Combines F-Regression (statistical) and Random Forest (embedded) importance.
    """
    print(f"\nPerforming Hybrid Feature Selection (Top {k})...")
    
    # 1. Statistical (F-test)
    selector = SelectKBest(f_regression, k='all')
    selector.fit(X_train, y_train)
    stat_scores = pd.Series(selector.scores_, index=X_train.columns).fillna(0)
    
    # 2. Model-based (Random Forest)
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_scores = pd.Series(rf.feature_importances_, index=X_train.columns)
    
    # 3. Normalize and Combine
    stat_norm = (stat_scores - stat_scores.min()) / (stat_scores.max() - stat_scores.min() + 1e-6) # Add epsilon to prevent division by zero
    rf_norm = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min() + 1e-6)
    
    hybrid_score = (stat_norm + rf_norm) / 2
    selected_features = hybrid_score.nlargest(k).index.tolist()
    
    print(f"Selected Features: {selected_features}")
    return selected_features

# --- MODEL CLASS ---
class DengueEnsemble:
    def __init__(self):
        self.models = {
            'RF': RandomForestRegressor(n_estimators=300, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
            'XGB': xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1),
            'GBM': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE)
        }
        self.weights = {'RF': 0.4, 'XGB': 0.4, 'GBM': 0.2}
        
    def fit(self, X_train, y_train):
        print("\nTraining Ensemble Models...")
        for name, model in self.models.items():
            print(f" -> Fitting {name}...")
            model.fit(X_train, y_train)
            
    def predict(self, X_test):
        final_pred = np.zeros(len(X_test))
        total_weight = sum(self.weights.values())
        
        individual_preds = {}
        for name, model in self.models.items():
            pred = model.predict(X_test)
            pred = np.maximum(pred, 0)
            individual_preds[name] = pred
            final_pred += pred * (self.weights[name] / total_weight)
            
        return final_pred, individual_preds

# --- EVALUATION AND PLOTS ---

def evaluate_model_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculates comprehensive regression and outbreak metrics.
    """
    # Regression metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    non_zero_mask = y_true > 0
    if non_zero_mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask])
    else:
        mape = np.inf
        
    # Outbreak detection (using 75th percentile as threshold)
    outbreak_threshold = np.percentile(y_true, 75)
    y_outbreak_true = (y_true > outbreak_threshold).astype(int)
    y_outbreak_pred = (y_pred > outbreak_threshold).astype(int)
    
    outbreak_precision = precision_score(y_outbreak_true, y_outbreak_pred, zero_division=0)
    outbreak_recall = recall_score(y_outbreak_true, y_outbreak_pred, zero_division=0)
    outbreak_f1 = f1_score(y_outbreak_true, y_outbreak_pred, zero_division=0)
    
    metrics = {
        'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2,
        'Outbreak_Threshold': outbreak_threshold,
        'Outbreak_Precision': outbreak_precision,
        'Outbreak_Recall': outbreak_recall,
        'Outbreak_F1': outbreak_f1,
        'Predictions': y_pred
    }
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")
    print(f"MAPE: {mape*100:.1f}% (cases > 0 only)")
    print(f"Outbreak F1-Score: {outbreak_f1:.3f} (Threshold: {outbreak_threshold:.1f})")
    
    return metrics


def plot_results_6_panel(y_true, y_pred, dates, title, save_path):
    """
    Generates a 6-panel plot for comprehensive model diagnosis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Calculate residuals and basic metrics
    residuals = y_true - y_pred
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    non_zero_mask_plot = y_true > 0
    if non_zero_mask_plot.sum() > 0:
        mape_plot = mean_absolute_percentage_error(y_true[non_zero_mask_plot], y_pred[non_zero_mask_plot])
    else:
        mape_plot = np.inf
        
    # 1. Time Series Plot
    ax1 = axes[0, 0]
    ax1.plot(dates, y_true, 'o-', label='Actual', color='#1f77b4', markersize=3, alpha=0.7)
    ax1.plot(dates, y_pred, 's--', label='Predicted', color='#ff7f0e', markersize=3, alpha=0.7)
    ax1.set_title('Time Series: Actual vs Predicted', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Dengue Cases')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Scatter Plot (Actual vs Predicted)
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred, alpha=0.6, color='darkgreen', s=20)
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction Line')
    ax2.set_title('Actual vs Predicted', fontsize=12)
    ax2.set_xlabel('Actual Cases')
    ax2.set_ylabel('Predicted Cases')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals Plot (Residuals vs Predicted)
    ax3 = axes[0, 2]
    ax3.scatter(y_pred, residuals, alpha=0.6, color='purple', s=20)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_title('Residuals vs Predicted', fontsize=12)
    ax3.set_xlabel('Predicted Cases')
    ax3.set_ylabel('Residuals')
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals Distribution
    ax4 = axes[1, 0]
    ax4.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_title('Distribution of Residuals', fontsize=12)
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.axvline(x=0, color='red', linestyle='--')
    ax4.grid(True, alpha=0.3)
    
    # 5. Boxplot by Value Range (Predictions over Quartiles of Actual Values)
    ax5 = axes[1, 1]
    try:
        bins = pd.qcut(y_true, q=4, labels=['Q1 (Low)', 'Q2 (Mid-Low)', 'Q3 (Mid-High)', 'Q4 (High)'], duplicates='drop')
        unique_bins = bins.unique()
        if len(unique_bins) >= 2:
            boxplot_data = [y_pred[bins == label] for label in unique_bins if (bins == label).any()]
            ax5.boxplot(boxplot_data, labels=[str(label) for label in unique_bins])
            ax5.set_title('Predictions by Quartile of Actual Values', fontsize=12)
        else:
            raise ValueError("Not enough unique bins")
    except (ValueError, TypeError):
        # Fallback if quantile binning fails (e.g., too many zero values)
        ax5.scatter(y_true, y_pred, alpha=0.5)
        ax5.set_title('Scatter: Actual vs Predicted (Fallback)', fontsize=12)
        ax5.set_xlabel('Actual Values')
        ax5.set_ylabel('Predictions')

    ax5.set_xlabel('Ranges/Quartiles')
    ax5.set_ylabel('Predictions')
    ax5.grid(True, alpha=0.3)
    
    # 6. Metrics Text Box
    ax6 = axes[1, 2]
    ax6.axis('off')

    metrics_text = f"""
PERFORMANCE METRICS

RMSE: {rmse:.2f}
MAE:  {mae:.2f}
R²:   {r2:.3f}
MAPE: {mape_plot*100:.1f}% (cases > 0 only)

Correlation: {np.corrcoef(y_true, y_pred)[0,1]:.3f}
Outliers (>2σ residuals): {np.sum(np.abs(residuals) > 2*np.std(residuals))}
Max Error: {np.max(np.abs(residuals)):.1f}
Zero Cases: {np.sum(y_true == 0)} ({np.mean(y_true == 0)*100:.1f}%)
"""
    
    ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Diagnostic Plot saved: {save_path}")
    plt.close()

def plot_results_time_series(y_test, y_pred, dates, title, save_path, rmse, r2):
    """
    Generates a Time Series plot comparing Actual vs Predicted values. (Simpler plot for final view)
    """
    plt.figure(figsize=(16, 8))
    
    # Plot Actual with shaded uncertainty (using residual std deviation)
    std_residuals = np.std(y_test - y_pred)
    plt.fill_between(dates, y_pred - std_residuals, y_pred + std_residuals, 
                     alpha=0.2, color='#ff7f0e', label=r'Prediction $\pm 1\sigma$')
    
    # Plot Actual
    sns.lineplot(x=dates, y=y_test, label='Actual', color='#1f77b4', 
                 marker='o', markersize=5, linewidth=1.5, alpha=0.8)
    
    # Plot Predicted
    sns.lineplot(x=dates, y=y_pred, label='Predicted', color='#ff7f0e', 
                 marker='x', markersize=6, linewidth=1.5, linestyle='--', alpha=0.9)
    
    plt.title(f"{title}\n(RMSE: {rmse:.2f} | R²: {r2:.3f})", fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Dengue Cases', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Time Series Plot saved successfully at: {save_path}")
    plt.close()

# --- MAIN PIPELINE ---
def main():
    models_dir, data_dir = ensure_dirs()
    file_path = data_dir / "cleaned_monthly_dengue_cases_final.csv"
    
    # 1. Load Data
    processor = DengueDataProcessor(file_path)
    df = processor.load_and_clean()
    
    if df is None:
        return

    # 2. Split & Impute
    X_train, X_test, y_train, y_test, dates_test, all_features = processor.prepare_split(df)
    
    # 3. Feature Selection
    selected_feats = hybrid_feature_selection(X_train, y_train, k=TOP_K_FEATURES)
    
    X_train_sel = X_train[selected_feats]
    X_test_sel = X_test[selected_feats]
    
    # 4. Train Model
    ensemble = DengueEnsemble()
    ensemble.fit(X_train_sel, y_train)
    
    # 5. Predict
    y_pred, individual_preds = ensemble.predict(X_test_sel)
    
    # 6. Evaluate
    metrics = evaluate_model_metrics(y_test, y_pred, "Ensemble")
    
    # 7. Visualize (Time Series - Simples)
    plot_results_time_series(
        y_test, y_pred, dates_test, 
        title="Dengue Forecasting - Ensemble Results (Temporal Split)", 
        save_path=models_dir / "forecast_vs_actual_simple.jpg",
        rmse=metrics['RMSE'], r2=metrics['R2']
    )
    
    # 8. Visualize (6 Panels - Diagnóstico Completo)
    plot_results_6_panel(
        y_test, y_pred, dates_test, 
        title="Dengue Forecasting - Ensemble Diagnostic Analysis",
        save_path=models_dir / "ensemble_diagnostic_analysis.jpg"
    )

    # 9. Save results
    results_df = pd.DataFrame({'date': dates_test, 'actual': y_test, 'predicted': y_pred})
    results_df.to_csv(data_dir / "forecast_results.csv", index=False, sep='\t')
    print(f"\nFinal results saved to: {data_dir}")


if __name__ == "__main__":
    main()