import pandas as pd
import os
from pathlib import Path

def clean_data():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    data_dir.mkdir(exist_ok=True)

    input_file = data_dir / "monthly_dengue_cases.csv"
    output_file = data_dir / "cleaned_monthly_dengue_cases.csv"

    try:
        df = pd.read_csv(input_file, sep=';')
        print(f"File uploaded successfully at {input_file}")
    except FileNotFoundError:
        print("Error: The file 'monthly_dengue_cases.csv' was not found. Please make sure it is in the same directory as your script or provide the full path.")
        exit()

    # before cleaning
    print("\n--- Initial Information of the DataFrame ---")
    df.info()

    print("\n--- First 5 rows of the DataFrame ---")
    print(df.head())

    print("\n--- Checking for missing values before cleaning ---")
    print(df.isnull().sum())


    # cleaning
    df_cleaned = df.copy()
    columns_to_check = ['precipitacao_total_mensal', 'temp_media_mensal', 'vento_vlc_media_mensal']

    # 1. Handling 'NULL' values as strings and converting to NaN.
    for col in columns_to_check:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].replace('NULL', pd.NA) 

    # 2. Conversion of the date column to datetime
    df_cleaned['dt_notificacao'] = pd.to_datetime(df_cleaned['dt_notificacao'], format='%d/%m/%Y', errors='coerce')
    
    # 3. Cleaning and converting numeric columns
    for col in columns_to_check:
        if col in df_cleaned.columns:
            # Remove thousand separators and replace commas with decimal points.
            df_cleaned[col] = df_cleaned[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            # Try to convert to numeric. Values that cannot be converted will become NaN.
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    # 4. Imputation of missing values for numerical climate columns
    print("\n--- Filling NaNs in the numeric columns using interpolation ---")
    for col in columns_to_check:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].interpolate(method='linear')
            df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
            df_cleaned[col] = df_cleaned[col].fillna(method='bfill')

    # 5. Verification and removal of duplicates
    print("\n--- Checking and removing duplicate lines (if any) ---")
    initial_rows_cleaned = df_cleaned.shape[0]
    df_cleaned.drop_duplicates(inplace=True)
    rows_after_duplicates_cleaned = df_cleaned.shape[0]
    if initial_rows_cleaned > rows_after_duplicates_cleaned:
        print(f"{initial_rows_cleaned - rows_after_duplicates_cleaned} been removed.")
    else:
        print("No duplicate line found.")

    # 6. Feature Engineering (Criação de Novas Variáveis a partir da data)
    print("\n--- Creating new climate features ---")
    # Remover linhas com datas inválidas (NaT) antes de extrair componentes
    df_cleaned = df_cleaned.dropna(subset=['dt_notificacao'])
    if not df_cleaned.empty:
        df_cleaned['year'] = df_cleaned['dt_notificacao'].dt.year
        df_cleaned['month'] = df_cleaned['dt_notificacao'].dt.month
        df_cleaned['day'] = df_cleaned['dt_notificacao'].dt.day
        df_cleaned['day_of_week'] = df_cleaned['dt_notificacao'].dt.dayofweek
        df_cleaned['week_of_year'] = df_cleaned['dt_notificacao'].dt.isocalendar().week.astype(int)
        df_cleaned['day_of_year'] = df_cleaned['dt_notificacao'].dt.dayofyear
    else:
        print("Attention: Empty DataFrame after removing invalid dates, could not create climate features.")


    # after cleaning
    print("\n--- Initial Information of the DataFrame after cleaning ---")
    df_cleaned.info()

    print("\n--- Checking for missing values after cleaning ---")
    print(df_cleaned.isnull().sum())

    print("\n--- First 5 rows of the DataFrame após cleaning ---")
    print(df_cleaned.head())

    try:
        df_cleaned.to_csv(output_file, index=False, sep=';', encoding='utf-8')
        return df_cleaned
    except Exception as e:
        print(f"\nError saving file: {str(e)}")
        return None

clean_data()