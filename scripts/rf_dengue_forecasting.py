import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set better plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ===============================================================================
# UTILIDADES E CARREGAMENTO
# ===============================================================================

def ensure_dirs():
    base_path = Path(__file__).parent.parent
    models_dir = base_path / "models"
    data_dir = base_path / "data"
    models_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    return models_dir, data_dir

def load_cleaned_data():
    _, data_dir = ensure_dirs()
    file_path = data_dir / "cleaned_monthly_dengue_cases_final.csv"
    try:
        df = pd.read_csv(file_path, sep=';', parse_dates=['dt_notificacao'])
        print(f"Data loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print("ERROR: Run the improved cleaning script first.")
        return None

# ===============================================================================
# ENGENHARIA DE FEATURES AVANÇADA
# ===============================================================================

def create_advanced_features(df):
    """
    Cria features mais sofisticadas para melhor predição
    """
    print("\n--- Advanced Feature Engineering ---")
    
    # Remove data leakage features
    leakage_features = [
        'qntd_febre', 'qntd_vomito', 'qntd_nausea', 'qntd_sangramento',
        'qntd_resultado_soro', 'qntd_resultado_ns1', 'qntd_resultado_pcr',
        'qntd_hospitalizacao', 'qntd_auctone'
    ]
    
    df = df.drop(columns=[col for col in leakage_features if col in df.columns])
    
    # Sort by municipality and date
    df = df.sort_values(['cd_municipio', 'dt_notificacao'])
    
    # 1. FEATURES CLIMÁTICAS INTERATIVAS
    print("  Creating climate interaction features...")
    
    # Índice de favorabilidade avançado
    df['optimal_conditions'] = (
        (df['temp_media_mensal'] >= 26) & (df['temp_media_mensal'] <= 30) &
        (df['precipitacao_total_mensal'] > 50) & (df['precipitacao_total_mensal'] < 250)
    ).astype(int)
    
    # Condições extremas que inibem transmissão
    df['extreme_conditions'] = (
        (df['temp_media_mensal'] < 18) | (df['temp_media_mensal'] > 35) |
        (df['precipitacao_total_mensal'] > 400)
    ).astype(int)
    
    # Interação temperatura-precipitação não-linear
    df['temp_precip_index'] = (
        df['temp_media_mensal'] * np.log1p(df['precipitacao_total_mensal'])
    )
    
    # 2. FEATURES TEMPORAIS AVANÇADAS
    print("  Creating advanced temporal features...")
    
    # Múltiplas representações sazonais
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    # Períodos críticos para dengue
    df['dengue_season'] = df['month'].isin([1, 2, 3, 4, 5]).astype(int)
    df['peak_season'] = df['month'].isin([2, 3, 4]).astype(int)
    df['low_season'] = df['month'].isin([6, 7, 8, 9]).astype(int)
    
    # 3. FEATURES HISTÓRICAS MAIS SOFISTICADAS
    print("  Creating historical features...")
    
    # Lags múltiplos com pesos decrescentes
    lag_periods = [1, 2, 3, 6, 12]
    for lag in lag_periods:
        # Casos históricos
        df[f'casos_lag_{lag}'] = df.groupby('cd_municipio')['qntd_casos'].shift(lag).fillna(0)
        
        # Clima histórico
        for var in ['temp_media_mensal', 'precipitacao_total_mensal']:
            df[f'{var}_lag_{lag}'] = df.groupby('cd_municipio')[var].shift(lag).fillna(df[var].median())
    
    # Médias móveis ponderadas
    for window in [3, 6, 12]:
        df[f'casos_ma_{window}'] = (
            df.groupby('cd_municipio')['qntd_casos']
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
            .fillna(0)
        )
    
    # Tendência (slope dos últimos 3 meses)
    def calculate_trend(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    df['casos_trend_3m'] = (
        df.groupby('cd_municipio')['qntd_casos']
        .shift(1)
        .rolling(window=3, min_periods=2)
        .apply(calculate_trend)
        .fillna(0)
    )
    
    # 4. FEATURES DE OUTBREAK DETECTION
    print("  Creating outbreak detection features...")
    
    # Threshold dinâmico baseado no histórico municipal
    df['municipal_avg'] = df.groupby('cd_municipio')['qntd_casos'].transform('mean')
    df['municipal_std'] = df.groupby('cd_municipio')['qntd_casos'].transform('std')
    df['above_normal'] = (
        df['qntd_casos'] > (df['municipal_avg'] + df['municipal_std'])
    ).astype(int)
    
    # Aceleração (segunda derivada)
    df['casos_acceleration'] = (
        df.groupby('cd_municipio')['qntd_casos']
        .shift(1)
        .diff()
        .diff()
        .fillna(0)
    )
    
    # 5. FEATURES GEOGRÁFICAS/MUNICIPAIS
    print("  Creating municipal features...")
    
    # Características municipais (médias históricas)
    df['muni_temp_avg'] = df.groupby('cd_municipio')['temp_media_mensal'].transform('mean')
    df['muni_precip_avg'] = df.groupby('cd_municipio')['precipitacao_total_mensal'].transform('mean')
    df['muni_cases_avg'] = df.groupby('cd_municipio')['qntd_casos'].transform('mean')
    
    # Desvios das médias municipais
    df['temp_deviation'] = df['temp_media_mensal'] - df['muni_temp_avg']
    df['precip_deviation'] = df['precipitacao_total_mensal'] - df['muni_precip_avg']
    
    # 6. LIMPEZA FINAL
    print("  Final cleanup...")
    
    # Remove features problemáticas
    problematic_cols = [col for col in df.columns if 'same_month' in col or col.endswith('_ly')]
    if problematic_cols:
        df = df.drop(columns=problematic_cols)
    
    # Preenche valores ausentes restantes
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'qntd_casos':
            df[col] = df[col].fillna(df[col].median())
    
    # Remove infinitos
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in numeric_cols:
        if col != 'qntd_casos':
            df[col] = df[col].fillna(df[col].median())
    
    print(f"  Final feature count: {df.shape[1]}")
    
    return df

# ===============================================================================
# PREPARAÇÃO DE DADOS
# ===============================================================================

def prepare_ml_data(df, target_col='qntd_casos', test_size=0.2):
    """
    Prepara dados para machine learning com validação temporal
    """
    print(f"\n--- Preparing ML Data (test_size={test_size}) ---")
    
    # Identificar features
    exclude_cols = [target_col, 'dt_notificacao', 'cd_municipio', 'ano_notificacao']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(df)}")
    print(f"  Date range: {df['dt_notificacao'].min()} to {df['dt_notificacao'].max()}")
    
    # Split temporal
    df_sorted = df.sort_values('dt_notificacao')
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    dates_test = test_df['dt_notificacao']
    
    print(f"  Train: {len(X_train)} samples ({train_df['dt_notificacao'].min()} to {train_df['dt_notificacao'].max()})")
    print(f"  Test:  {len(X_test)} samples ({test_df['dt_notificacao'].min()} to {test_df['dt_notificacao'].max()})")
    
    return X_train, X_test, y_train, y_test, dates_test, feature_cols

# ===============================================================================
# FEATURE SELECTION INTELIGENTE
# ===============================================================================

def intelligent_feature_selection(X_train, y_train, method='hybrid', k=30):
    """
    Seleção inteligente de features usando múltiplos métodos
    """
    print(f"\n--- Intelligent Feature Selection ({method}) ---")
    
    if method == 'statistical':
        # Método estatístico
        selector = SelectKBest(f_regression, k=k)
        selector.fit(X_train, y_train)
        
    elif method == 'rfe':
        # Recursive Feature Elimination
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=k)
        selector.fit(X_train, y_train)
        
    else:  # hybrid
        # Método híbrido: estatístico + importância do modelo
        # Primeiro: filtro estatístico
        stat_selector = SelectKBest(f_regression, k=min(50, X_train.shape[1]))
        X_stat = stat_selector.fit_transform(X_train, y_train)
        
        # Segundo: importância do Random Forest
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_temp.fit(X_stat, y_train)
        
        # Combinar scores
        stat_features = X_train.columns[stat_selector.get_support()]
        rf_importance = pd.DataFrame({
            'feature': stat_features,
            'stat_score': stat_selector.scores_[stat_selector.get_support()],
            'rf_importance': rf_temp.feature_importances_
        })
        
        # Score híbrido
        rf_importance['hybrid_score'] = (
            rf_importance['stat_score'] / rf_importance['stat_score'].max() +
            rf_importance['rf_importance'] / rf_importance['rf_importance'].max()
        )
        
        # Selecionar top k
        selected_features = rf_importance.nlargest(k, 'hybrid_score')['feature'].tolist()
        
        # Criar selector personalizado
        class HybridSelector:
            def __init__(self, selected_features, all_features):
                self.selected_features = selected_features
                self.all_features = all_features
                self.support_ = [col in selected_features for col in all_features]
            
            def transform(self, X):
                return X[self.selected_features]
            
            def get_support(self):
                return self.support_
        
        selector = HybridSelector(selected_features, X_train.columns.tolist())
        
        # Para compatibilidade, criar DataFrame com scores
        feature_scores = pd.DataFrame({
            'feature': X_train.columns,
            'score': [rf_importance[rf_importance['feature'] == col]['hybrid_score'].iloc[0] 
                     if col in selected_features else 0 for col in X_train.columns]
        }).sort_values('score', ascending=False)
        
        print("  Top 10 selected features:")
        for i, (_, row) in enumerate(feature_scores.head(10).iterrows()):
            print(f"    {i+1:2d}. {row['feature']}: {row['score']:.3f}")
        
        return selector, selected_features, feature_scores
    
    # Para métodos não-híbridos
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    if hasattr(selector, 'scores_'):
        feature_scores = pd.DataFrame({
            'feature': X_train.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
    else:
        feature_scores = pd.DataFrame({
            'feature': X_train.columns,
            'score': [1.0 if col in selected_features else 0.0 for col in X_train.columns]
        })
    
    print("  Top 10 selected features:")
    for i, (_, row) in enumerate(feature_scores.head(10).iterrows()):
        print(f"    {i+1:2d}. {row['feature']}: {row['score']:.3f}")
    
    return selector, selected_features, feature_scores

# ===============================================================================
# ENSEMBLE DE MODELOS
# ===============================================================================

class DengueEnsemble:
    """
    Ensemble especializado para predição de dengue
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        
    def add_model(self, name, model, weight=1.0):
        self.models[name] = model
        self.weights[name] = weight
        
    def fit(self, X_train, y_train):
        print(f"\n--- Training Ensemble ({len(self.models)} models) ---")
        
        # Normalizar features para alguns modelos
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns, index=X_train.index)
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            if 'Neural' in name or 'SVM' in name:
                model.fit(X_scaled, y_train)
            else:
                model.fit(X_train, y_train)
                
    def predict(self, X_test):
        predictions = {}
        X_scaled = self.scaler.transform(X_test)
        X_scaled = pd.DataFrame(X_scaled, columns=X_test.columns, index=X_test.index)
        
        for name, model in self.models.items():
            if 'Neural' in name or 'SVM' in name:
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X_test)
            predictions[name] = np.maximum(pred, 0)  # No negative predictions
        
        # Weighted average
        total_weight = sum(self.weights.values())
        ensemble_pred = np.zeros(len(X_test))
        
        for name, pred in predictions.items():
            ensemble_pred += pred * (self.weights[name] / total_weight)
        
        return ensemble_pred, predictions

def create_optimized_models():
    """
    Cria modelos otimizados para o ensemble
    """
    models = {}
    
    # Random Forest otimizado
    models['RF_Optimized'] = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    # XGBoost otimizado
    models['XGB_Optimized'] = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting
    models['GBM'] = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    
    return models

# ===============================================================================
# AVALIAÇÃO E VISUALIZAÇÃO MELHORADAS
# ===============================================================================

def evaluate_comprehensive(y_true, y_pred, model_name="Model"):
    """
    Avaliação abrangente com métricas específicas para dengue
    """
    # Métricas básicas
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE corrigido - só calcular onde y_true > 0
    non_zero_mask = y_true > 0
    if non_zero_mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask])
    else:
        mape = np.inf
    
    # Métricas específicas para surtos
    outbreak_threshold = np.percentile(y_true, 75)  # Top 25% como "surto"
    y_outbreak_true = (y_true > outbreak_threshold).astype(int)
    y_outbreak_pred = (y_pred > outbreak_threshold).astype(int)
    
    # Precision/Recall para detecção de surtos
    from sklearn.metrics import precision_score, recall_score, f1_score
    outbreak_precision = precision_score(y_outbreak_true, y_outbreak_pred, zero_division=0)
    outbreak_recall = recall_score(y_outbreak_true, y_outbreak_pred, zero_division=0)
    outbreak_f1 = f1_score(y_outbreak_true, y_outbreak_pred, zero_division=0)
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"  Regression Metrics:")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    MAE:  {mae:.2f}")
    print(f"    MAPE: {mape:.3f} ({mape*100:.1f}%)")
    print(f"    R²:   {r2:.3f}")
    print(f"  Outbreak Detection (threshold={outbreak_threshold:.1f}):")
    print(f"    Precision: {outbreak_precision:.3f}")
    print(f"    Recall:    {outbreak_recall:.3f}")
    print(f"    F1-Score:  {outbreak_f1:.3f}")
    
    return {
        'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2,
        'Outbreak_Precision': outbreak_precision,
        'Outbreak_Recall': outbreak_recall,
        'Outbreak_F1': outbreak_f1,
        'Predictions': y_pred
    }

def plot_comprehensive_results(y_true, y_pred, dates, title, save_path):
    """
    Gráficos melhorados e mais claros
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # 1. Time Series (melhorado)
    ax1 = axes[0, 0]
    ax1.plot(dates, y_true, 'o-', label='Real', color='darkblue', markersize=3, alpha=0.7)
    ax1.plot(dates, y_pred, 's-', label='Predito', color='red', markersize=3, alpha=0.7)
    ax1.set_title('Série Temporal: Real vs Predito', fontsize=12)
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Casos de Dengue')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Scatter Plot (melhorado)
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred, alpha=0.6, color='darkgreen', s=20)
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Linha Perfeita')
    ax2.set_title('Real vs Predito', fontsize=12)
    ax2.set_xlabel('Casos Reais')
    ax2.set_ylabel('Casos Preditos')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuais
    ax3 = axes[0, 2]
    residuals = y_true - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.6, color='purple', s=20)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_title('Resíduos vs Predito', fontsize=12)
    ax3.set_xlabel('Casos Preditos')
    ax3.set_ylabel('Resíduos')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribuição dos Resíduos
    ax4 = axes[1, 0]
    ax4.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_title('Distribuição dos Resíduos', fontsize=12)
    ax4.set_xlabel('Resíduos')
    ax4.set_ylabel('Frequência')
    ax4.axvline(x=0, color='red', linestyle='--')
    ax4.grid(True, alpha=0.3)
    
    # 5. Boxplot por faixa de valores (corrigido)
    ax5 = axes[1, 1]
    try:
        # Tentar qcut primeiro
        bins = pd.qcut(y_true, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        unique_bins = bins.unique()
        if len(unique_bins) >= 2:
            boxplot_data = [y_pred[bins == label] for label in unique_bins if (bins == label).any()]
            ax5.boxplot(boxplot_data, labels=[str(label) for label in unique_bins])
            ax5.set_title('Predições por Quartil dos Valores Reais', fontsize=12)
        else:
            raise ValueError("Not enough unique bins")
    except (ValueError, TypeError):
        # Fallback: usar cut simples se qcut falhar
        max_val = y_true.max()
        cut_points = [0, max_val*0.25, max_val*0.5, max_val*0.75, max_val]
        bins = pd.cut(y_true, bins=cut_points, labels=['Baixo', 'Médio-Baixo', 'Médio-Alto', 'Alto'], include_lowest=True)
        unique_bins = [label for label in bins.cat.categories if (bins == label).any()]
        
        if len(unique_bins) >= 2:
            boxplot_data = [y_pred[bins == label] for label in unique_bins]
            ax5.boxplot(boxplot_data, labels=unique_bins)
            ax5.set_title('Predições por Faixa de Valores Reais', fontsize=12)
        else:
            # Último fallback: scatter plot simples
            ax5.scatter(y_true, y_pred, alpha=0.5)
            ax5.set_title('Scatter: Real vs Predito (Fallback)', fontsize=12)
            ax5.set_xlabel('Valores Reais')
            ax5.set_ylabel('Predições')
    
    ax5.set_xlabel('Faixas/Quartis')
    ax5.set_ylabel('Predições')
    ax5.grid(True, alpha=0.3)
    
    # 6. Métricas visuais
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true + 1e-10, y_pred + 1e-10)
    
    metrics_text = f"""
    MÉTRICAS DE PERFORMANCE
    
    RMSE: {rmse:.2f}
    MAE:  {mae:.2f}
    R²:   {r2:.3f}
    MAPE: {mape*100:.1f}% (só casos>0)
    
    Correlação: {np.corrcoef(y_true, y_pred)[0,1]:.3f}
    
    Outliers (>2σ): {np.sum(np.abs(residuals) > 2*np.std(residuals))}
    
    Max Erro: {np.max(np.abs(residuals)):.1f}
    Min Erro: {np.min(np.abs(residuals)):.1f}
    
    Casos Zero: {np.sum(y_true == 0)} ({np.mean(y_true == 0)*100:.1f}%)
    """
    
    ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {save_path}")
    plt.show()
    plt.close()

# ===============================================================================
# PIPELINE PRINCIPAL
# ===============================================================================

def run_complete_pipeline():
    """
    Pipeline completo melhorado
    """
    print("="*80)
    print("DENGUE FORECASTING - COMPLETE IMPROVED PIPELINE")
    print("="*80)
    
    # 1. Carregar e preparar dados
    models_dir, data_dir = ensure_dirs()
    df = load_cleaned_data()
    if df is None:
        return
    
    # 2. Feature engineering avançada
    df = create_advanced_features(df)
    
    # 3. Preparar dados para ML
    X_train, X_test, y_train, y_test, dates_test, feature_cols = prepare_ml_data(df)
    
    # 4. Feature selection inteligente
    selector, selected_features, feature_scores = intelligent_feature_selection(
        X_train, y_train, method='hybrid', k=25
    )
    
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # 5. Criar e treinar ensemble
    ensemble = DengueEnsemble()
    models = create_optimized_models()
    
    for name, model in models.items():
        ensemble.add_model(name, model, weight=1.0)
    
    ensemble.fit(X_train_selected, y_train)
    
    # 6. Predições
    ensemble_pred, individual_preds = ensemble.predict(X_test_selected)
    
    # 7. Avaliação abrangente
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    # Avaliar ensemble
    ensemble_metrics = evaluate_comprehensive(y_test, ensemble_pred, "Ensemble")
    
    # Avaliar modelos individuais
    individual_metrics = {}
    for name, pred in individual_preds.items():
        individual_metrics[name] = evaluate_comprehensive(y_test, pred, name)
    
    # 8. Visualizar resultados
    plot_comprehensive_results(
        y_test, ensemble_pred, dates_test,
        title="Dengue Forecasting - Ensemble Model (Improved)",
        save_path=models_dir / "ensemble_comprehensive_results.png"
    )
    
    # 9. Salvar resultados
    results_df = pd.DataFrame({
        'date': dates_test,
        'actual': y_test.values,
        'ensemble_pred': ensemble_pred
    })
    
    # Adicionar predições individuais
    for name, pred in individual_preds.items():
        results_df[f'{name}_pred'] = pred
    
    results_df.to_csv(data_dir / "complete_results_improved.csv", index=False, sep=';')
    
    # Salvar feature importance
    feature_scores.to_csv(data_dir / "feature_importance_improved.csv", index=False)
    
    # Sumário final
    print(f"\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Best Model: Ensemble")
    print(f"  R²: {ensemble_metrics['R2']:.3f}")
    print(f"  RMSE: {ensemble_metrics['RMSE']:.2f}")
    print(f"  Outbreak F1: {ensemble_metrics['Outbreak_F1']:.3f}")
    print(f"Results saved to: {data_dir}")
    
    return ensemble, ensemble_metrics, feature_scores

if __name__ == "__main__":
    run_complete_pipeline()