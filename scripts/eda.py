import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional

# --- Project Structure and Utilities ---

def get_project_root() -> Path:
    """Returns the project root directory, assuming the script is in 'scripts/'."""
    return Path(__file__).parent.parent

def ensure_models_dir(project_root: Path) -> Path:
    """Ensure that the 'models' directory exists for saving plots."""
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir

def load_cleaned_data(project_root: Path) -> Optional[pd.DataFrame]:
    """Loads the cleaned data from the specified path, using the correct separator and encoding."""
    data_dir = project_root / "data"
    file_path = data_dir / "cleaned_monthly_dengue_cases_final.csv"

    try:
        # CRITICAL FIX: Using tab ('\t') as separator and 'latin-1' encoding
        df = pd.read_csv(
            file_path, 
            sep='\t', 
            encoding='latin-1',
            parse_dates=['dt_notificacao'],
            dtype={'cd_municipio': str} 
        )
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV (Check separator/name/encoding): {e}")
        return None

# --- Plotting Functions ---

def plot_dengue_cases_over_time(df: pd.DataFrame, models_dir: Path):
    """Plots dengue cases aggregated over time."""
    plt.figure(figsize=(14, 5))
    df_monthly = df.groupby('dt_notificacao')['qntd_casos'].sum().reset_index()
    sns.lineplot(data=df_monthly, x='dt_notificacao', y='qntd_casos', color='darkblue', linewidth=1.5)
    plt.title('Dengue Cases Over Time (Monthly Sum)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Cases', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plot_path = models_dir / "dengue_cases_over_time.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()
    
def plot_symptoms_barchart(df: pd.DataFrame, models_dir: Path):
    """Plots a bar chart showing the prevalence of each symptom as a percentage of total cases."""
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
    symptom_prevalence = {name: (count / total_cases) * 100 for name, count in symptom_counts.items()}

    symptoms_df = pd.DataFrame(list(symptom_prevalence.items()), columns=['Symptom', 'Prevalence (%)'])
    symptoms_df = symptoms_df.sort_values('Prevalence (%)', ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x='Prevalence (%)', y='Symptom', data=symptoms_df, palette='rocket', orient='h'
    )

    plt.title('Symptom Prevalence in Dengue Cases', fontsize=16)
    plt.xlabel('Prevalence (%)', fontsize=12)
    plt.ylabel('Symptom', fontsize=12)
    plt.xlim(0, max(symptoms_df['Prevalence (%)']) * 1.1)

    for index, value in enumerate(symptoms_df['Prevalence (%)']):
        ax.text(value + 0.5, index, f'{value:.1f}%', color='black', ha="left", va="center")

    plt.tight_layout()
    plot_path = models_dir / "symptoms_barchart.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, models_dir: Path):
    """Plots the correlation between current cases and climate variables."""
    plt.figure(figsize=(8, 6))
    cols = [
        'qntd_casos', 'precipitacao_total_mensal',
        'temp_media_mensal', 'vento_vlc_media_mensal'
    ]

    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, linecolor='black')
    plt.title('Correlation between Cases and Climate Variables', fontsize=14)
    plt.tight_layout()

    plot_path = models_dir / "correlation_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()


def plot_cases_by_month(df: pd.DataFrame, models_dir: Path):
    """Plots the distribution (boxplot) of dengue cases by month."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='month', y='qntd_casos', data=df, palette='Spectral')
    plt.title('Distribution of Dengue Cases by Month (Seasonality)', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Cases', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = models_dir / "cases_by_month.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()


def plot_seasonality_heatmap(df: pd.DataFrame, models_dir: Path):
    """Plots a heatmap showing monthly cases per year to visualize long-term seasonality."""
    if 'year' not in df.columns or 'month' not in df.columns:
         print("Columns 'year' or 'month' missing for seasonality heatmap.")
         return
         
    df_pivot = df.groupby(['year', 'month'])['qntd_casos'].sum().unstack()
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        df_pivot, annot=True, fmt='.0f', cmap='YlGnBu', 
        linewidths=0.5, linecolor='gray'
    )
    plt.title("Monthly Dengue Cases per Year", fontsize=16)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Year", fontsize=12)
    plt.tight_layout()
    
    plot_path = models_dir / "seasonality_heatmap.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    plt.close()


def plot_climatic_scatter(df: pd.DataFrame, models_dir: Path, x_col: str, y_col: str):
    """Plots a scatter plot with regression line between a climate variable and dengue cases."""
    
    plt.figure(figsize=(8, 6))
    sns.regplot(
        x=x_col, 
        y=y_col, 
        data=df, 
        scatter_kws={'alpha':0.4, 's':20}, 
        line_kws={"color": "red"}
    )
    
    # Title formatting improvement
    title = f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}"
    plt.title(title, fontsize=14)
    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    plt.tight_layout()

    plot_path = models_dir / f"scatter_{y_col}_vs_{x_col}.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    plt.close()

# --- Main Execution Block ---
if __name__ == "__main__":
    
    project_root = get_project_root()
    models_dir = ensure_models_dir(project_root)
    
    df = load_cleaned_data(project_root)
    
    if df is not None:
        # 1. Essential Time-Series Pre-processing
        df.sort_values(by=['cd_municipio', 'dt_notificacao'], inplace=True)
        
        # 2. Time Feature Creation
        df['year'] = df['dt_notificacao'].dt.year
        df['month'] = df['dt_notificacao'].dt.month
        
        print("\n--- Generating Plots (EDA) ---")
        
        # 3. Plotting Execution
        plot_dengue_cases_over_time(df, models_dir)
        plot_symptoms_barchart(df, models_dir)
        plot_correlation_heatmap(df, models_dir)
        plot_cases_by_month(df, models_dir)
        plot_seasonality_heatmap(df, models_dir)
        
        # New scatter analyses
        plot_climatic_scatter(df, models_dir, 'temp_media_mensal', 'qntd_casos')
        plot_climatic_scatter(df, models_dir, 'precipitacao_total_mensal', 'qntd_casos')