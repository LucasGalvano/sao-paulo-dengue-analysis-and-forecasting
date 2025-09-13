import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def ensure_models_dir():
    """Ensure that model diretory exists"""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def load_cleaned_data():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    file_path = data_dir / "cleaned_monthly_dengue_cases.csv"

    try:
        df = pd.read_csv(file_path, sep=';', parse_dates=['dt_notificacao'])
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print("File not found.")
        return None
    

def plot_dengue_cases_over_time(df, models_dir):
    plt.figure(figsize=(14, 5))
    df_monthly = df.groupby('dt_notificacao')['qntd_casos'].sum().reset_index()
    sns.lineplot(data=df_monthly, x='dt_notificacao', y='qntd_casos')
    plt.title('Dengue Cases Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.tight_layout()

    plot_path = models_dir / "dengue_cases_over_time.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()
    plt.close()


def plot_symptoms_barchart(df, models_dir):
    """
    Plots a bar chart showing the prevalence of each symptom as a percentage of total cases.
    This is more appropriate than a pie chart because symptoms are not mutually exclusive.
    """
    symptom_cols = {
        'qntd_febre': 'Fever',
        'qntd_vomito': 'Vomit',
        'qntd_nausea': 'Nausea',
        'qntd_sangramento': 'Bleeding'
    }
    
    total_cases = df['qntd_casos'].sum()
    if total_cases == 0:
        print("No dengue cases to analyze symptoms for.")
        return

    symptom_counts = {name: df[col].sum() for col, name in symptom_cols.items()}
    
    # Calculate prevalence percentage for each symptom
    symptom_prevalence = {name: (count / total_cases) * 100 for name, count in symptom_counts.items()}

    # Create a DataFrame for plotting
    symptoms_df = pd.DataFrame(list(symptom_prevalence.items()), columns=['Symptom', 'Prevalence (%)'])
    symptoms_df = symptoms_df.sort_values('Prevalence (%)', ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x='Prevalence (%)',
        y='Symptom',
        data=symptoms_df,
        palette='viridis',
        orient='h'
    )

    plt.title('Symptom Prevalence in Dengue Cases', fontsize=16)
    plt.xlabel('Prevalence (%)', fontsize=12)
    plt.ylabel('Symptom', fontsize=12)
    plt.xlim(0, max(symptoms_df['Prevalence (%)']) * 1.1)

    # Add percentage labels to the bars
    for index, value in enumerate(symptoms_df['Prevalence (%)']):
        ax.text(value + 0.5, index, f'{value:.1f}%', color='black', ha="left", va="center")

    plt.tight_layout()
    
    # Save the plot
    plot_path = models_dir / "symptoms_barchart.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()
    plt.close()


def plot_correlation_heatmap(df, models_dir):
    plt.figure(figsize=(8, 6))
    cols = [
        'qntd_casos',
        'precipitacao_total_mensal',
        'temp_media_mensal',
        'vento_vlc_media_mensal'
    ]

    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation between Cases and Climate Variables')
    plt.tight_layout()

    plot_path = models_dir / "correlation_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()
    plt.close()


def plot_cases_by_month(df, models_dir):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='month', y='qntd_casos', data=df)
    plt.title('Distribution of Dengue Cases by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Cases')
    plt.tight_layout()

    plot_path = models_dir / "cases_by_month.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()
    plt.close()


def plot_seasonality_heatmap(df, models_dir):
    df_pivot = df.groupby(['year', 'month'])['qntd_casos'].sum().unstack()
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_pivot, annot=True, fmt='.0f', cmap='YlOrRd')
    plt.title("Monthly Dengue Cases per Year")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    
    plot_path = models_dir / "seasonality_heatmap.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    plt.show()
    plt.close()


def add_lag_features(df, models_dir, lags=[3, 4]):
    """
    Creates lagged features for cases and climate variables using a groupby
    approach to ensure correct time-series relationships per municipality.
    """
    print("\n--- Creating lagged features with groupby() ---")
    df_lagged = df.copy()
    
    # Create lag features correctly using groupby
    for lag in lags:
        df_lagged[f'qntd_casos_lag{lag}'] = df_lagged.groupby('cd_municipio')['qntd_casos'].shift(lag)
        df_lagged[f'precipitacao_lag{lag}'] = df_lagged.groupby('cd_municipio')['precipitacao_total_mensal'].shift(lag)
        df_lagged[f'temp_media_lag{lag}'] = df_lagged.groupby('cd_municipio')['temp_media_mensal'].shift(lag)
        df_lagged[f'vento_media_lag{lag}'] = df_lagged.groupby('cd_municipio')['vento_vlc_media_mensal'].shift(lag)
    
    # Print info to check for NaNs before plotting
    print("\nDataFrame info after adding lagged features:")
    df_lagged.info()
    print("\nNumber of NaNs after adding lagged features:")
    print(df_lagged.isnull().sum())
    
    # Plot of the correlation matrix
    cols_corr = [
        'qntd_casos',
        'precipitacao_total_mensal',
        'temp_media_mensal',
        'vento_vlc_media_mensal'
    ] + [col for col in df_lagged.columns if 'lag' in col]
    
    df_corr = df_lagged[cols_corr].dropna().copy()
    
    print("\nPlotting correlation matrix. NaNs introduced by shift() are temporarily dropped for the plot.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df_corr.corr(numeric_only=True),
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 8}
    )
    plt.xticks(rotation=45, ha='right')
    plt.title("Correlation matrix (Relevant Features)", pad=20)
    plt.tight_layout()

    plot_path = models_dir / "lag_features_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()
    plt.close()

    return df_lagged


if __name__ == "__main__":
    df = load_cleaned_data()
    if df is not None:
        models_dir = ensure_models_dir()
        df.sort_values(by=['cd_municipio', 'dt_notificacao'], inplace=True)
        plot_dengue_cases_over_time(df, models_dir)
        plot_symptoms_barchart(df, models_dir)
        plot_correlation_heatmap(df, models_dir)
        plot_cases_by_month(df, models_dir)
        plot_seasonality_heatmap(df, models_dir)
        df_with_lags = add_lag_features(df, models_dir, lags=[3, 4])