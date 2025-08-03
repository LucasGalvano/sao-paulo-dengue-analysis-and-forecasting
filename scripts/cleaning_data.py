import pandas as pd
from pathlib import Path


def get_project_paths():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "monthly_dengue_cases.csv", data_dir / "cleaned_monthly_dengue_cases.csv"


def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, sep=';')
        print(f"File uploaded successfully at {file_path}")
        return df
    except FileNotFoundError:
        print("Error: The file 'monthly_dengue_cases.csv' was not found.")
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
            df[col] = df[col].replace('NULL', pd.NA)
    return df


def convert_columns(df, columns):
    df['dt_notificacao'] = pd.to_datetime(df['dt_notificacao'], format='%d/%m/%Y', errors='coerce')
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def fill_missing_values(df, columns):
    print("\n--- Filling NaNs in the numeric columns using interpolation ---")
    for col in columns:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    return df


def remove_duplicates(df):
    print("\n--- Checking and removing duplicate lines (if any) ---")
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed = initial_rows - df.shape[0]
    print(f"{removed} duplicate rows removed." if removed else "No duplicate line found.")
    return df


def create_date_features(df):
    print("\n--- Creating new climate features ---")
    df = df.dropna(subset=['dt_notificacao'])
    if not df.empty:
        df['year'] = df['dt_notificacao'].dt.year
        df['month'] = df['dt_notificacao'].dt.month
        df['day'] = df['dt_notificacao'].dt.day
        df['day_of_week'] = df['dt_notificacao'].dt.dayofweek
        df['week_of_year'] = df['dt_notificacao'].dt.isocalendar().week.astype(int)
        df['day_of_year'] = df['dt_notificacao'].dt.dayofyear
    else:
        print("Attention: Empty DataFrame after removing invalid dates.")
    return df


def show_final_info(df):
    print("\n--- Information after cleaning ---")
    df.info()
    print("\n--- Missing values after cleaning ---")
    print(df.isnull().sum())
    print("\n--- First 5 rows after cleaning ---")
    print(df.head())


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

    columns_to_clean = ['precipitacao_total_mensal', 'temp_media_mensal', 'vento_vlc_media_mensal']
    df = handle_null_strings(df, columns_to_clean)
    df = convert_columns(df, columns_to_clean)
    df = fill_missing_values(df, columns_to_clean)
    df = remove_duplicates(df)
    df = create_date_features(df)

    show_final_info(df)

    return save_dataframe(df, output_file)


if __name__ == "__main__":
    clean_data()
