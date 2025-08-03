import pandas as pd
import numpy as np
from pathlib import Path


def get_project_paths():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "monthly_dengue_cases.tab", data_dir / "cleaned_monthly_dengue_cases.csv"


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


def show_initial_info(df):
    print("\n--- Initial Information of the DataFrame ---")
    df.info()
    print("\n--- First 5 rows of the DataFrame ---")
    print(df.head())
    print("\n--- Checking for missing values before cleaning ---")
    print(df.isnull().sum())


def handle_null_strings(df, columns):
    for col in columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace('NULL', pd.NA)
    return df


def handle_suspicious_zeros(df):
    print("\n--- Converting suspicious zeros to NaNs ---")
    if 'temp_media_mensal' in df.columns:
        initial_zeros_temp = (df['temp_media_mensal'] == 0).sum()
        if initial_zeros_temp > 0:
            df['temp_media_mensal'].replace(0, np.nan, inplace=True)
            print(f"Converted {initial_zeros_temp} '0' values in 'temp_media_mensal' to NaN.")
        else:
            print("No '0' values found in 'temp_media_mensal' to convert.")
    return df


def fill_missing_values(df, columns):
    print("\n--- Filling NaNs in the numeric columns using interpolation ---")
    for col in columns:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')
            df[col] = df[col].ffill().bfill()
    return df


def remove_duplicates(df):
    print("\n--- Checking and removing duplicate lines (if any) ---")
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed = initial_rows - df.shape[0]
    print(f"{removed} duplicate rows removed." if removed else "No duplicate line found.")
    return df


def create_date_features(df):
    print("\n--- Creating new date features ---")
    df['dt_notificacao'] = pd.to_datetime(df['dt_notificacao'], format='%Y-%m-%d', errors='coerce')
    initial_rows = df.shape[0]
    df.dropna(subset=['dt_notificacao'], inplace=True)
    removed_dates = initial_rows - df.shape[0]
    if removed_dates > 0:
        print(f"Removed {removed_dates} rows with invalid/missing 'dt_notificacao'.")

    if not df.empty:
        df['year'] = df['dt_notificacao'].dt.year
        df['month'] = df['dt_notificacao'].dt.month
        df['day'] = df['dt_notificacao'].dt.day
        df['day_of_week'] = df['dt_notificacao'].dt.dayofweek
        df['week_of_year'] = df['dt_notificacao'].dt.isocalendar().week.astype(int)
        df['day_of_year'] = df['dt_notificacao'].dt.dayofyear
    else:
        print("Attention: DataFrame is empty after removing invalid dates.")
    return df


def create_has_features(df):
    print("\n--- Creating 'has_' binary features ---")

    climate_cols = ['precipitacao_total_mensal', 'vento_vlc_media_mensal']
    for col in climate_cols:
        if col in df.columns:
            new_col_name = f'has_{col.replace("_total_mensal", "").replace("_media_mensal", "")}'
            df[new_col_name] = (df[col] > 0).astype(int)
            print(f"Created '{new_col_name}' feature.")


    highly_sparse_qntd_cols = [
        'qntd_hospitalizacao', 'qntd_resultado_soro', 'qntd_resultado_ns1', 
        'qntd_resultado_pcr', 'qntd_auctone', 'qntd_febre', 
        'qntd_vomito', 'qntd_nausea', 'qntd_sangramento'
    ]
    
    for col in highly_sparse_qntd_cols:
        if col in df.columns:
            new_col_name = f'has_{col.replace("qntd_", "")}'
            df[new_col_name] = (df[col] > 0).astype(int)
            print(f"Created '{new_col_name}' feature.")
            
    return df


def show_final_info(df):
    print("\n--- Information after cleaning ---")
    df.info()
    print("\n--- Missing values after cleaning ---")
    print(df.isnull().sum())
    print("\n--- First 5 rows after cleaning ---")
    print(df.head())

    columns_for_zero_check = [
        'qntd_casos', 'qntd_hospitalizacao', 'qntd_resultado_soro',
        'qntd_resultado_ns1', 'qntd_resultado_pcr', 'qntd_auctone',
        'qntd_febre', 'qntd_vomito', 'qntd_nausea', 'qntd_sangramento',
        'precipitacao_total_mensal', 'temp_media_mensal', 'vento_vlc_media_mensal'
    ]
    check_zero_percentage(df, columns_for_zero_check)


def check_zero_percentage(df, columns_to_check):
    print("\n--- Percentage of Zero Values in Selected Columns (After Cleaning) ---")
    total_rows = len(df)
    if total_rows == 0:
        print("DataFrame is empty, cannot calculate zero percentage.")
        return

    for col in columns_to_check:
        if col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_numeric_dtype(df[col]):
                zero_count = (df[col] == 0).sum()
                zero_percentage = (zero_count / total_rows) * 100
                print(f"- Column '{col}': {zero_percentage:.2f}% zeros ({zero_count}/{total_rows})")
            else:
                print(f"- Column '{col}' is not numeric and skipped for zero check.")
        else:
            print(f"- Column '{col}' not found in DataFrame.")
    print("-" * 50)


def save_dataframe(df, output_path):
    try:
        df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
        print(f"\nFile saved successfully at {output_path}")
        return df
    except Exception as e:
        print(f"\nError saving file: {str(e)}")
        return None


def clean_data():
    input_file, output_file = get_project_paths()
    df = read_csv_file(input_file)

    show_initial_info(df)

    columns_for_null_handling_strings = ['precipitacao_total_mensal', 'temp_media_mensal', 'vento_vlc_media_mensal']
    df = handle_null_strings(df, columns_for_null_handling_strings)
    df = handle_suspicious_zeros(df)
    columns_for_filling_nans = ['precipitacao_total_mensal', 'temp_media_mensal', 'vento_vlc_media_mensal']
    df = fill_missing_values(df, columns_for_filling_nans)
    df = remove_duplicates(df)
    df = create_date_features(df)
    df = create_has_features(df)

    show_final_info(df)

    return save_dataframe(df, output_file)


if __name__ == "__main__":
    clean_data()