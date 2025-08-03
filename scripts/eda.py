import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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
        print(" Data loaded successfully.")
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



def plot_symptoms_pie_chart(df, models_dir):
    # Calculate total cases
    total_cases = df['qntd_casos'].sum()
    if total_cases == 0:
        print("No cases found to plot.")
        return
    
    # Calculate cases with each symptom
    fever_cases = df['qntd_febre'].sum()
    vomiting_cases = df['qntd_vomito'].sum()
    nausea_cases = df['qntd_nausea'].sum()
    bleeding_cases = df['qntd_sangramento'].sum()
    
    # Calculate percentages
    fever_percent = (fever_cases / total_cases) * 100
    vomiting_percent = (vomiting_cases / total_cases) * 100
    nausea_percent = (nausea_cases / total_cases) * 100
    bleeding_percent = (bleeding_cases / total_cases) * 100
    
    # Calculate "Other Symptoms" (avoid negative values)
    other_percent = max(0, 100 - (fever_percent + vomiting_percent + nausea_percent + bleeding_percent))
    
    # Filter out near-zero percentages (< 0.1%) to avoid tiny slices
    labels = []
    sizes = []
    colors = []
    color_palette = ['#ff9999', '#66b3ff', '#99ff99', '#faa41a', '#a62a7e']
    
    if fever_percent >= 0.1:
        labels.append('Fever')
        sizes.append(fever_percent)
        colors.append(color_palette[0])
    if vomiting_percent >= 0.1:
        labels.append('Vomit')
        sizes.append(vomiting_percent)
        colors.append(color_palette[1])
    if nausea_percent >= 0.1:
        labels.append('Nausea')
        sizes.append(nausea_percent)
        colors.append(color_palette[2])
    if bleeding_percent >= 0.1:
        labels.append('Bleeding')
        sizes.append(bleeding_percent)
        colors.append(color_palette[3])
    if other_percent >= 0.1:
        labels.append('Other Symptoms')
        sizes.append(other_percent)
        colors.append(color_palette[4])
    
    # If all are <0.1%, show at least the main symptoms
    if not sizes:
        print("All symptom percentages are very low (<0.1%). Adjusting thresholds...")
        labels = ['Fever', 'Vomit', 'Nausea', 'Bleeding', 'Other Symptoms']
        sizes = [fever_percent, vomiting_percent, nausea_percent, bleeding_percent, other_percent]
        colors = color_palette

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct=lambda p: f'{p:.1f}%' if p >= 0.1 else '', startangle=90)
    plt.title('Distribution of Symptoms in Dengue Cases')
    
    plot_path = models_dir / "symptoms_pie_chart.png"
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



def add_lag_features(df, models_dir, lags=[1, 2]):
    df_lagged = df.copy()
    
    # Create lag features
    for lag in lags:
        df_lagged[f'qntd_casos_lag{lag}'] = df_lagged['qntd_casos'].shift(lag)
        df_lagged[f'precipitacao_lag{lag}'] = df_lagged['precipitacao_total_mensal'].shift(lag)
        df_lagged[f'temp_media_lag{lag}'] = df_lagged['temp_media_mensal'].shift(lag)
        df_lagged[f'vento_media_lag{lag}'] = df_lagged['vento_vlc_media_mensal'].shift(lag)
    
    # Plot of the correlation matrix
    cols_corr = [
        'qntd_casos',
        'precipitacao_total_mensal',
        'temp_media_mensal',
        'vento_vlc_media_mensal'
    ] + [col for col in df_lagged.columns if 'lag' in col]
    
    df_corr = df_lagged[cols_corr].copy()
    
    
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
        plot_dengue_cases_over_time(df, models_dir)
        plot_symptoms_pie_chart(df, models_dir)
        plot_correlation_heatmap(df, models_dir)
        plot_cases_by_month(df, models_dir)
        add_lag_features(df, models_dir, lags=[1, 2])