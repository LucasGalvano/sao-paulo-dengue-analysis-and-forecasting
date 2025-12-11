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
import xgboost as xgb
import warnings

# --- SETUP ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# --- CONSTANTS ---
LEAKAGE_FEATURES = [
    'qntd_febre', 'qntd_vomito', 'qntd_nausea', 'qntd_sangramento',
    'qntd_resultado_soro', 'qntd_resultado_ns1', 'qntd_resultado_pcr',
    'qntd_hospitalizacao', 'qntd_auctone'
]

TEST_SIZE = 0.2
TOP_K_FEATURES = 25
RANDOM_STATE = 42
N_SPLITS = 5 

def ensure_dirs():
    """Creates necessary directories."""
    base_path = Path(__file__).parent.parent
    models_dir = base_path / "models"
    data_dir = base_path / "data"
    models_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    return models_dir, data_dir



# DATA PROCESSING CLASS (FIXED)
class DengueDataProcessor:
    """
    Handles data loading, cleaning, and splitting with proper temporal validation.
    
    FIXES:
    - Removed separate imputer (now uses median fill consistently)
    - Fixed dimensionality issues
    - Added proper NaN handling before feature selection
    """
    
    def __init__(self, file_path, target_col='qntd_casos'):
        self.file_path = file_path
        self.target_col = target_col
        
    def load_and_clean(self):
        """Loads data and removes leakage features immediately."""
        try:
            df = pd.read_csv(
                self.file_path, 
                sep='\t', 
                encoding='latin-1', 
                parse_dates=['dt_notificacao']
            )
            print(f"Data loaded successfully: {df.shape}")
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return None

        # 1. Drop rows with missing Target
        initial_len = len(df)
        df.dropna(subset=[self.target_col], inplace=True)
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"   Dropped {dropped} rows with missing target")
        
        # 2. Sort by date (Crucial for Time Series)
        df = df.sort_values('dt_notificacao').reset_index(drop=True)
        
        # 3. Remove Leakage Features
        cols_to_drop = [c for c in LEAKAGE_FEATURES if c in df.columns]
        if cols_to_drop:
            print(f"Removing {len(cols_to_drop)} leakage features: {cols_to_drop}")
            df.drop(columns=cols_to_drop, inplace=True)
            
        return df

    def prepare_split(self, df, test_size=TEST_SIZE):
        """
        Splits data chronologically and handles preprocessing correctly.
        
        FIXED:
        - Remove constant/all-NaN features BEFORE imputation
        - Use consistent median imputation (no SimpleImputer object)
        - Ensure feature_cols match transformed data
        """
        # 1. Define Candidate Features
        exclude_cols = [
            self.target_col, 'dt_notificacao', 'cd_municipio', 
            'ano_notificacao', 'month', 'quarter'
        ]
        candidate_cols = [
            c for c in df.columns 
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        
        print(f"\n{'='*80}")
        print("DATA PREPARATION")
        print(f"{'='*80}")
        
        # 2. Identify and remove constant/all-NaN features FIRST
        print("Checking for problematic features...")
        problematic_cols = []
        
        for col in candidate_cols:
            if df[col].isnull().all():
                problematic_cols.append(col)
                print(f" {col}: All NaN")
            elif df[col].nunique(dropna=True) <= 1:
                problematic_cols.append(col)
                print(f" {col}: Constant (zero variance)")
        
        if problematic_cols:
            print(f"   Dropping {len(problematic_cols)} problematic features")
            candidate_cols = [c for c in candidate_cols if c not in problematic_cols]
        
        # 3. Temporal Split
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"\nTemporal Split:")
        print(f"   Train: {train_df['dt_notificacao'].min().date()} to "
              f"{train_df['dt_notificacao'].max().date()} ({len(train_df):,} samples)")
        print(f"   Test:  {test_df['dt_notificacao'].min().date()} to "
              f"{test_df['dt_notificacao'].max().date()} ({len(test_df):,} samples)")
        
        # 4. Extract features and target
        X_train = train_df[candidate_cols].copy()
        X_test = test_df[candidate_cols].copy()
        y_train = train_df[self.target_col].copy()
        y_test = test_df[self.target_col].copy()
        dates_test = test_df['dt_notificacao'].copy()
        
        # 5. CONSISTENT IMPUTATION (fit on train, apply to test)
        print(f"\n Imputing missing values (median strategy)...")
        print(f"   Features before imputation: {len(candidate_cols)}")
        
        # Calculate medians from TRAIN set only
        train_medians = X_train.median()
        
        # Fill NaN with train medians
        X_train_imputed = X_train.fillna(train_medians)
        X_test_imputed = X_test.fillna(train_medians)
        
        # Verify no NaNs remain
        assert not X_train_imputed.isnull().any().any(), "NaNs remain in X_train after imputation!"
        assert not X_test_imputed.isnull().any().any(), "NaNs remain in X_test after imputation!"
        
        print(f"Imputation complete (no NaNs remain)")
        
        return X_train_imputed, X_test_imputed, y_train, y_test, dates_test, candidate_cols



# FEATURE SELECTION
def hybrid_feature_selection(X_train, y_train, k=TOP_K_FEATURES):
    """
    Combines F-Regression (statistical) and Random Forest (embedded) importance.
    """
    print(f"\n{'='*80}")
    print(f"FEATURE SELECTION (Hybrid Method, Top {k})")
    print(f"{'='*80}")
    
    # 1. Statistical (F-test)
    selector = SelectKBest(f_regression, k='all')
    selector.fit(X_train, y_train)
    stat_scores = pd.Series(selector.scores_, index=X_train.columns).fillna(0)
    
    # 2. Model-based (Random Forest)
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_scores = pd.Series(rf.feature_importances_, index=X_train.columns)
    
    # 3. Normalize and Combine (add small epsilon to avoid division by zero)
    stat_norm = (stat_scores - stat_scores.min()) / (stat_scores.max() - stat_scores.min() + 1e-10)
    rf_norm = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min() + 1e-10)
    
    hybrid_score = (stat_norm + rf_norm) / 2
    
    # 4. Select top k
    top_features = hybrid_score.nlargest(k)
    selected_features = top_features.index.tolist()
    
    # 5. Display results
    print(f"\nTop {min(10, k)} Selected Features:")
    for i, (feat, score) in enumerate(top_features.head(10).items(), 1):
        print(f"   {i:2d}. {feat:40s} (score: {score:.4f})")
    
    return selected_features, hybrid_score



# ENSEMBLE MODEL (FIXED WEIGHTS)
class DengueEnsemble:
    """
    Ensemble of RF, XGB, and GBM with equal weighting (simplified).
    
    FIXED: Using equal weights (1/3 each) instead of arbitrary weights.
    """
    
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=300, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE, 
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=500, 
                learning_rate=0.05, 
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE, 
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200, 
                learning_rate=0.1, 
                max_depth=5,
                subsample=0.8,
                random_state=RANDOM_STATE
            )
        }
        # FIXED: Equal weights (more transparent)
        self.weights = {name: 1/len(self.models) for name in self.models}
        
    def fit(self, X_train, y_train):
        print(f"\n{'='*80}")
        print("TRAINING ENSEMBLE")
        print(f"{'='*80}")
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
        print("All models trained successfully")
            
    def predict(self, X_test):
        final_pred = np.zeros(len(X_test))
        individual_preds = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_test)
            pred = np.maximum(pred, 0)  # No negative predictions
            individual_preds[name] = pred
            final_pred += pred * self.weights[name]
            
        return final_pred, individual_preds



# CROSS-VALIDATION
def temporal_cross_validation(X, y, n_splits=N_SPLITS):
    """
    Performs TimeSeriesSplit cross-validation to assess model stability.
    
    Returns average metrics across folds.
    """
    print(f"\n{'='*80}")
    print(f"TEMPORAL CROSS-VALIDATION ({n_splits} folds)")
    print(f"{'='*80}")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = {'R2': [], 'RMSE': [], 'MAE': []}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train simple RF for CV (faster than full ensemble)
        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = np.maximum(model.predict(X_val_cv), 0)
        
        # Calculate metrics
        r2 = r2_score(y_val_cv, y_pred_cv)
        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
        mae = mean_absolute_error(y_val_cv, y_pred_cv)
        
        cv_scores['R2'].append(r2)
        cv_scores['RMSE'].append(rmse)
        cv_scores['MAE'].append(mae)
        
        print(f"   Fold {fold}: R²={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
    
    # Average results
    avg_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
    std_scores = {metric: np.std(scores) for metric, scores in cv_scores.items()}
    
    print(f"\n Cross-Validation Results (Average ± Std):")
    print(f"   R²:   {avg_scores['R2']:.3f} ± {std_scores['R2']:.3f}")
    print(f"   RMSE: {avg_scores['RMSE']:.2f} ± {std_scores['RMSE']:.2f}")
    print(f"   MAE:  {avg_scores['MAE']:.2f} ± {std_scores['MAE']:.2f}")
    
    return avg_scores, std_scores



# EVALUATION
def evaluate_comprehensive(y_true, y_pred, model_name="Model"):
    """
    Calculates ALL metrics including MAPE and Outbreak detection.
    """
    # Regression metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (only for non-zero actual values)
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
        'RMSE': rmse, 
        'MAE': mae, 
        'MAPE': mape, 
        'R2': r2,
        'Outbreak_Threshold': outbreak_threshold,
        'Outbreak_Precision': outbreak_precision,
        'Outbreak_Recall': outbreak_recall,
        'Outbreak_F1': outbreak_f1,
        'Predictions': y_pred
    }
    
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} EVALUATION")
    print(f"{'='*80}")
    print(f"   Regression Metrics:")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE:  {mae:.2f}")
    print(f"   R²:   {r2:.3f}")
    print(f"   MAPE: {mape*100:.1f}% (cases > 0 only)")
    print(f"\n Outbreak Detection (threshold = {outbreak_threshold:.1f} cases):")
    print(f"   Precision: {outbreak_precision:.3f}")
    print(f"   Recall:    {outbreak_recall:.3f}")
    print(f"   F1-Score:  {outbreak_f1:.3f}")
    
    return metrics


def evaluate_baseline(y_train, y_test):
    """
    Evaluates naive baseline models for comparison.
    """
    print(f"\n{'='*80}")
    print("BASELINE COMPARISON")
    print(f"{'='*80}")
    
    # Baseline 1: Historical mean
    mean_pred = np.full(len(y_test), y_train.mean())
    mean_rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
    mean_r2 = r2_score(y_test, mean_pred)
    
    print(f"   Naive Mean Baseline:")
    print(f"   RMSE: {mean_rmse:.2f}")
    print(f"   R²:   {mean_r2:.3f}")
    
    return {'Mean_RMSE': mean_rmse, 'Mean_R2': mean_r2}



# PLOTTING 
def plot_results_diagnostic(y_true, y_pred, dates, save_path):
    """
    Generates comprehensive 6-panel diagnostic plot.
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle('Dengue Forecasting - Ensemble Diagnostic Analysis', fontsize=16, y=0.98)
    
    residuals = y_true - y_pred
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    non_zero_mask = y_true > 0
    mape = mean_absolute_percentage_error(
        y_true[non_zero_mask], y_pred[non_zero_mask]
    ) if non_zero_mask.sum() > 0 else np.inf
    
    # 1. Time Series
    ax1 = axes[0, 0]
    ax1.plot(dates, y_true, 'o-', label='Actual', color='#1f77b4', markersize=3, alpha=0.7)
    ax1.plot(dates, y_pred, 's--', label='Predicted', color='#ff7f0e', markersize=3, alpha=0.7)
    ax1.set_title('Time Series: Actual vs Predicted', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Dengue Cases')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Scatter Plot
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred, alpha=0.6, color='darkgreen', s=20)
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_title('Actual vs Predicted', fontsize=12)
    ax2.set_xlabel('Actual Cases')
    ax2.set_ylabel('Predicted Cases')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals Plot
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
    
    # 5. Boxplot by Quartile
    ax5 = axes[1, 1]
    try:
        bins = pd.qcut(y_true, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        unique_bins = bins.unique()
        if len(unique_bins) >= 2:
            boxplot_data = [y_pred[bins == label] for label in unique_bins]
            ax5.boxplot(boxplot_data, labels=[str(label) for label in unique_bins])
            ax5.set_title('Predictions by Actual Value Quartile', fontsize=12)
        else:
            raise ValueError("Not enough unique bins")
    except (ValueError, TypeError):
        ax5.scatter(y_true, y_pred, alpha=0.5)
        ax5.set_title('Scatter: Actual vs Predicted', fontsize=12)
    
    ax5.set_xlabel('Quartiles')
    ax5.set_ylabel('Predictions')
    ax5.grid(True, alpha=0.3)
    
    # 6. Metrics Text Box
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    metrics_text = f"""
PERFORMANCE METRICS

RMSE: {rmse:.2f}
MAE:  {mae:.2f}
R²:   {r2:.3f}
MAPE: {mape*100:.1f}%

Correlation: {np.corrcoef(y_true, y_pred)[0,1]:.3f}
Outliers (>2σ): {np.sum(np.abs(residuals) > 2*np.std(residuals))}
Max Error: {np.max(np.abs(residuals)):.1f}
Zero Cases: {np.sum(y_true == 0)} ({np.mean(y_true == 0)*100:.1f}%)
"""
    
    ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nDiagnostic plot saved: {save_path}")
    plt.close()


def plot_results_simple(y_true, y_pred, dates, save_path, rmse, r2):
    """
    Generates simple time series plot.
    """
    plt.figure(figsize=(16, 8))
    
    std_residuals = np.std(y_true - y_pred)
    plt.fill_between(dates, y_pred - std_residuals, y_pred + std_residuals, 
                     alpha=0.2, color='#ff7f0e', label=r'Prediction $\pm 1\sigma$')
    
    sns.lineplot(x=dates, y=y_true, label='Actual', color='#1f77b4', 
                 marker='o', markersize=5, linewidth=1.5, alpha=0.8)
    
    sns.lineplot(x=dates, y=y_pred, label='Predicted', color='#ff7f0e', 
                 marker='x', markersize=6, linewidth=1.5, linestyle='--', alpha=0.9)
    
    plt.title(f"Dengue Forecasting - Ensemble Results\n(RMSE: {rmse:.2f} | R²: {r2:.3f})", 
              fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Dengue Cases', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Time series plot saved: {save_path}")
    plt.close()



# MAIN PIPELINE


def main():
    """
    Complete training pipeline with all fixes and improvements.
    """
    print("\n" + "="*80)
    print("DENGUE FORECASTING - FIXED & VALIDATED PIPELINE")
    print("="*80)
    
    models_dir, data_dir = ensure_dirs()
    file_path = data_dir / "cleaned_monthly_dengue_cases_final.csv"
    
    # 1. Load Data
    processor = DengueDataProcessor(file_path)
    df = processor.load_and_clean()
    
    if df is None:
        print("\nPipeline aborted: Could not load data")
        return
    
    # 2. Split & Prepare
    X_train, X_test, y_train, y_test, dates_test, all_features = processor.prepare_split(df)
    
    # 3. Feature Selection
    selected_feats, feature_scores = hybrid_feature_selection(X_train, y_train, k=TOP_K_FEATURES)
    
    X_train_sel = X_train[selected_feats]
    X_test_sel = X_test[selected_feats]
    
    # 4. Cross-Validation
    cv_scores, cv_std = temporal_cross_validation(X_train_sel, y_train, n_splits=N_SPLITS)
    
    # 5. Baseline Comparison
    baseline_metrics = evaluate_baseline(y_train, y_test)
    
    # 6. Train Final Ensemble
    ensemble = DengueEnsemble()
    ensemble.fit(X_train_sel, y_train)
    
    # 7. Predict
    y_pred, individual_preds = ensemble.predict(X_test_sel)
    
    # 8. Evaluate (COMPLETE METRICS)
    metrics = evaluate_comprehensive(y_test, y_pred, "Ensemble")
    
    # 9. Compare with Baseline
    print(f"\n{'='*80}")
    print("IMPROVEMENT OVER BASELINE")
    print(f"{'='*80}")
    print(f"   Baseline RMSE: {baseline_metrics['Mean_RMSE']:.2f}")
    print(f"   Ensemble RMSE: {metrics['RMSE']:.2f}")
    print(f"   Improvement:   {baseline_metrics['Mean_RMSE'] - metrics['RMSE']:.2f} "
          f"({(1 - metrics['RMSE']/baseline_metrics['Mean_RMSE'])*100:.1f}%)")
    
    # 10. Visualize
    plot_results_simple(
        y_test, y_pred, dates_test,
        save_path=models_dir / "forecast_vs_actual.jpg",
        rmse=metrics['RMSE'], r2=metrics['R2']
    )
    
    plot_results_diagnostic(
        y_test, y_pred, dates_test,
        save_path=models_dir / "ensemble_diagnostic_analysis.jpg"
    )
    
    # 11. Save Results
    results_df = pd.DataFrame({
        'date': dates_test,
        'actual': y_test.values,
        'predicted': y_pred
    })
    results_df.to_csv(data_dir / "forecast_results.csv", index=False, sep='\t')
    
    # Save feature importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_scores.index,
        'importance': feature_scores.values
    }).sort_values('importance', ascending=False)
    feature_importance_df.to_csv(data_dir / "feature_importance.csv", index=False, sep='\t')
    
    print(f"\nResults saved to: {data_dir}")
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()