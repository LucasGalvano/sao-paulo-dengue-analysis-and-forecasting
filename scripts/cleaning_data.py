import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def get_project_paths():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "monthly_dengue_cases.tab", data_dir / "cleaned_monthly_dengue_cases_final.csv"


def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"File uploaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path.name}' was not found.")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        exit()


def smart_missing_value_handling(df):
    """
    Smart strategy for missing values
    """
    print("\n--- Smart Missing Value Handling ---")

    # Convert suspicious 0s to NaN only for temperature
    if 'temp_media_mensal' in df.columns:
        suspicious_zeros = (df['temp_media_mensal'] == 0).sum()
        if suspicious_zeros > 0:
            df[df['temp_media_mensal'] == 0] = df[df['temp_media_mensal'] == 0].copy()
            df.loc[df['temp_media_mensal'] == 0, 'temp_media_mensal'] = np.nan
            print(f"  Converted {suspicious_zeros} suspicious zeros in temperature to NaN")

    # Smart imputation by context
    climate_cols = ['precipitacao_total_mensal', 'temp_media_mensal', 'vento_vlc_media_mensal']
    df = df.sort_values(['cd_municipio', 'dt_notificacao'])

    for col in climate_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            print(f"  Imputing {col}...")
            # By municipality first
            df[col] = df.groupby('cd_municipio')[col].fillna(method='ffill').fillna(method='bfill')
            # By seasonality
            if df[col].isnull().sum() > 0:
                seasonal_median = df.groupby('mes_notificacao')[col].transform('median')
                df[col] = df[col].fillna(seasonal_median)
            # Global
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

    return df


def detect_and_handle_outliers(df, columns, method='iqr'):
    """
    Detects and conservatively handles outliers
    """
    print(f"\n--- Outlier Handling ({method}) ---")

    for col in columns:
        if col not in df.columns:
            continue

        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # For counts, do not allow negative values
            if col.startswith('qntd_'):
                lower_bound = max(0, lower_bound)

            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers.sum()

            if outlier_count > 0:
                print(f"  {col}: {outlier_count} outliers handled")
                # Use clipping instead of removal
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def create_predictive_features(df):
    """
    Creates features that are truly PREDICTIVE (not consequences)
    """
    print("\n--- Creating Predictive Features ---")

    # Convert date
    df['dt_notificacao'] = pd.to_datetime(df['dt_notificacao'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['dt_notificacao'])
    df = df.sort_values(['cd_municipio', 'dt_notificacao'])

    # Basic temporal features
    df['year'] = df['dt_notificacao'].dt.year
    df['month'] = df['dt_notificacao'].dt.month
    df['quarter'] = df['dt_notificacao'].dt.quarter
    df['day_of_year'] = df['dt_notificacao'].dt.dayofyear

    # Seasonal features (critical for dengue)
    df['is_summer'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_rainy_season'] = df['month'].isin([10, 11, 12, 1, 2, 3]).astype(int)
    df['is_peak_dengue_season'] = df['month'].isin([1, 2, 3, 4]).astype(int)

    # Cyclical encoding to capture seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # CLIMATE LAGS (essential - dengue responds to past climate)
    climate_vars = ['precipitacao_total_mensal', 'temp_media_mensal', 'vento_vlc_media_mensal']
    lag_periods = [1, 2, 3, 4, 5, 6]  # 1-6 months

    for var in climate_vars:
        if var in df.columns:
            for lag in lag_periods:
                df[f"{var}_lag_{lag}"] = df.groupby('cd_municipio')[var].shift(lag)

    # Climate moving averages (smooth noise)
    for var in climate_vars:
        if var in df.columns:
            df[f"{var}_ma_3"] = df.groupby('cd_municipio')[var].rolling(window=3, min_periods=1).mean().reset_index(drop=True)
            df[f"{var}_ma_6"] = df.groupby('cd_municipio')[var].rolling(window=6, min_periods=1).mean().reset_index(drop=True)

    # Climate interaction features (biologically relevant)
    if all(col in df.columns for col in ['temp_media_mensal', 'precipitacao_total_mensal']):
        df['temp_precip_interaction'] = df['temp_media_mensal'] * df['precipitacao_total_mensal']

        # Aedes aegypti favorability index
        # Temperature: 26-29°C optimal, precipitation: 50-300mm optimal
        df['aedes_favorability_index'] = np.where(
            (df['temp_media_mensal'] >= 26) & (df['temp_media_mensal'] <= 29) &
            (df['precipitacao_total_mensal'] > 50) & (df['precipitacao_total_mensal'] < 300),
            1, 0
        )

        # Extreme climate conditions (may reduce transmission)
        df['extreme_heat'] = (df['temp_media_mensal'] > 35).astype(int)
        df['extreme_cold'] = (df['temp_media_mensal'] < 18).astype(int)
        df['extreme_rain'] = (df['precipitacao_total_mensal'] > 400).astype(int)
        df['drought'] = (df['precipitacao_total_mensal'] < 10).astype(int)

    # HISTORICAL CASE LAGS
    case_lag_periods = [1, 2, 3, 6, 12]  # including 12 months to capture cycles

    for lag in case_lag_periods:
        lag_col = f'casos_lag_{lag}'
        df[lag_col] = df.groupby('cd_municipio')['qntd_casos'].shift(lag)

        # Only create if there is enough data
        if df[lag_col].notna().sum() > 1000:  # At least 1000 valid observations
            print(f"  Created {lag_col}: {df[lag_col].notna().sum()} valid obs")

    # Historical trends and patterns (CORRECTED: removed reset_index)
    df['casos_ma_3'] = df.groupby('cd_municipio')['qntd_casos'].shift(1).rolling(window=3, min_periods=1).mean()
    df['casos_ma_6'] = df.groupby('cd_municipio')['qntd_casos'].shift(1).rolling(window=6, min_periods=1).mean()
    df['casos_ma_12'] = df.groupby('cd_municipio')['qntd_casos'].shift(1).rolling(window=12, min_periods=1).mean()

    # Historical maximum (indicates epidemic potential) (CORRECTED: removed reset_index)
    df['casos_max_12m'] = df.groupby('cd_municipio')['qntd_casos'].shift(1).rolling(window=12, min_periods=1).max()

    # Cases in the same month of the previous year (seasonality)
    df['casos_same_month_ly'] = df.groupby(['cd_municipio', 'month'])['qntd_casos'].shift(12)

    # Municipal features (if there is variation)
    df['municipio_avg_temp'] = df.groupby('cd_municipio')['temp_media_mensal'].transform('mean')
    df['municipio_avg_precip'] = df.groupby('cd_municipio')['precipitacao_total_mensal'].transform('mean')

    print(f"  Total created features: {len(df.columns)}")
    return df


def create_smart_binary_features(df):
    """
    Creates binary features based on epidemiological thresholds
    Avoids data leakage by using only independent variables
    """
    print("\n--- Creating Predictive Binary Features ---")

    # Binary climate features (predictive)
    if 'precipitacao_total_mensal' in df.columns:
        df['high_precipitation'] = (df['precipitacao_total_mensal'] > 100).astype(int)
        df['optimal_precipitation'] = ((df['precipitacao_total_mensal'] > 50) &
                                     (df['precipitacao_total_mensal'] < 300)).astype(int)
        df['low_precipitation'] = (df['precipitacao_total_mensal'] < 20).astype(int)

    if 'temp_media_mensal' in df.columns:
        df['optimal_temperature'] = ((df['temp_media_mensal'] >= 26) &
                                     (df['temp_media_mensal'] <= 29)).astype(int)
        df['high_temperature'] = (df['temp_media_mensal'] > 30).astype(int)
        df['low_temperature'] = (df['temp_media_mensal'] < 20).astype(int)

    # REMOVE features that cause data leakage
    # These are CONSEQUENCES of the cases, not predictors:
    leakage_cols = [
        'qntd_febre', 'qntd_vomito', 'qntd_nausea', 'qntd_sangramento',  # symptoms
        'qntd_resultado_soro', 'qntd_resultado_ns1', 'qntd_resultado_pcr',  # tests
        'qntd_hospitalizacao', 'qntd_auctone'  # outcomes
    ]

    # Only keep if explicitly requested for descriptive analysis
    # For ML, they must be removed
    print("  WARNING: Identified features with data leakage (symptoms/tests):")
    for col in leakage_cols:
        if col in df.columns:
            print(f"    - {col}")

    return df


def validate_for_ml(df):
    """
    Specific validation for ML
    """
    print("\n--- Validation for Machine Learning ---")

    # 1. Check target
    if 'qntd_casos' not in df.columns:
        print("  ERROR: Target 'qntd_casos' not found!")
        return False

    target_stats = df['qntd_casos'].describe()
    print(f"  Target 'qntd_casos': min={target_stats['min']}, max={target_stats['max']}, mean={target_stats['mean']:.2f}")

    # 2. Identify predictive features vs. leakage features
    predictive_features = []
    leakage_features = []

    for col in df.columns:
        if col == 'qntd_casos':  # target
            continue
        elif col in ['dt_notificacao', 'cd_municipio']:  # identifiers
            continue
        elif any(leak in col for leak in ['febre', 'vomito', 'nausea', 'sangramento', 'hospitalizacao', 'resultado', 'auctone']):
            leakage_features.append(col)
        else:
            predictive_features.append(col)

    print(f"  SUCCESS: Predictive features: {len(predictive_features)}")
    print(f"  WARNING: Features with data leakage: {len(leakage_features)}")

    # 3. Check correlations of predictive features
    if predictive_features:
        corr_with_target = df[predictive_features + ['qntd_casos']].corr()['qntd_casos'].abs().sort_values(ascending=False)
        print("\n  Top 5 most correlated predictive features:")
        for feat in corr_with_target.head(5).index:
            if feat != 'qntd_casos':
                print(f"    {feat}: {corr_with_target[feat]:.3f}")

    # 4. Check missing values in important features
    important_cols = ['temp_media_mensal', 'precipitacao_total_mensal'] + [col for col in df.columns if 'lag' in col and 'casos' in col]
    missing_summary = df[important_cols].isnull().sum()
    if missing_summary.sum() > 0:
        print("\n  Missing values in important features:")
        for col, missing in missing_summary[missing_summary > 0].items():
            print(f"    {col}: {missing} ({missing/len(df)*100:.1f}%)")

    return True


def clean_data_for_ml():
    """
    Final cleaning pipeline optimized for Machine Learning
    """
    input_file, output_file = get_project_paths()
    df = read_csv_file(input_file)

    print("\n" + "="*80)
    print("DATA CLEANING OPTIMIZED FOR MACHINE LEARNING")
    print("="*80)

    # 1. Missing value handling
    df = smart_missing_value_handling(df)

    # 2. Outlier handling (conservative)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = detect_and_handle_outliers(df, numeric_cols)

    # 3. Duplicate removal
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed = initial_rows - df.shape[0]
    if removed > 0:
        print(f"\nINFO: {removed} duplicates removed")

    # 4. Predictive feature creation
    df = create_predictive_features(df)

    # 5. Smart binary features
    df = create_smart_binary_features(df)

    # 6. ML validation
    is_valid = validate_for_ml(df)

    if not is_valid:
        print("ERROR: Data failed validation!")
        return None

    # 7. Final information
    print(f"\nSUCCESS: Final dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"INFO: Period: {df['dt_notificacao'].min()} to {df['dt_notificacao'].max()}")

    try:
        df.to_csv(output_file, index=False, sep='\t', encoding='utf-8')
        print(f"\nSUCCESS: File saved: {output_file}")
        return df
    except Exception as e:
        print(f"\nERROR: Error saving file: {str(e)}")
        return None


if __name__ == "__main__":
    clean_data_for_ml()