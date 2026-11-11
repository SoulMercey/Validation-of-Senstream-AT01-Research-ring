import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Suppress matplotlib transparency warnings for EPS output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "Results"
GRAPHS_DIR = RESULTS_DIR / "Graphs"
THESIS_DIR = REPO_ROOT / "thesis" / "Slike"


def _decimal_comma(value, _position):
    """Formatira števila na oseh z decimalno vejico."""
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", ",")


def apply_decimal_comma(fig, x=True, y=True):
    """Nastavi slovenski zapis decimalnih števil na oseh grafa."""
    formatter = FuncFormatter(_decimal_comma)
    for ax in fig.axes:
        if x:
            ax.xaxis.set_major_formatter(formatter)
        if y:
            ax.yaxis.set_major_formatter(formatter)

def resolve_result_path(filename):
    candidate_paths = [RESULTS_DIR / filename]
    if filename == "cvx_quality_metrics.csv":
        candidate_paths.append(RESULTS_DIR / "cvxeda_quality_metrics.csv")
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename} in {RESULTS_DIR}")




def make_box_plots(column):
    if isinstance(column, str):
        columns = [column]
    else:
        columns = list(column)

    if not columns:
        raise ValueError("column must contain at least one column name")

    df_nk2 = pd.read_csv(resolve_result_path("nk2_quality_metrics.csv"))
    df_cvx = pd.read_csv(resolve_result_path("cvx_quality_metrics.csv"))
    df_manual = pd.read_csv(resolve_result_path("manual_quality_metrics.csv"))

    n_cols = len(columns)
    n_rows = max(1, int(np.ceil(n_cols / 2)))
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows), squeeze=False)
    flat_axes = axes.flatten()

    for idx, col in enumerate(columns):
        ax = flat_axes[idx]
        data = [df_nk2[col].dropna(), df_cvx[col].dropna(), df_manual[col].dropna()]
        ax.boxplot(data, tick_labels=['NK2', 'CVX', 'Manual'])
        ax.set_title(f"Box Plot of {col}")
        ax.set_ylabel(col)
        ax.set_xlabel("Methods")
        ax.tick_params(axis='x', rotation=45)

    for ax in flat_axes[n_cols:]:
        ax.axis('off')

    plt.tight_layout()
    #plt.show()
    #Manual filtering in NeuroKit2 sta pokazala zelo podobno porazdelitev napake fazične komponente. Metoda cvxEDA pa je dosegla višjo mediano MAE in bistveno večjo razpršenost rezultatov, kar kaže na manj stabilno delovanje med posameznimi udeleženci.



def plot_spearman_scatter(df, col, output_name):
    df_sorted = df.sort_values(by=col, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    # 's' controls the marker size, 'zorder' ensures points stay on top of the grid lines
    ax.scatter(df_sorted["segment"], df_sorted[col], color="dodgerblue", edgecolors="black", s=100, zorder=3)
    ax.grid(axis='both', linestyle='--', zorder=1)
    title_name = col.replace('_', ' ').title()
    ax.set_title(f"{title_name} by Segment", fontsize=14, fontweight='bold')
    ax.set_xlabel("Segment", fontsize=12)
    ax.set_ylabel(title_name, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    output = f"{GRAPHS_DIR}/{output_name}.pdf"
    plt.savefig(output, format="pdf")
    #plt.show()

    print(f"Scatter plot saved successfully as {output}")

def plot_peak_data(df, col, output_name):
    df_sorted = df.sort_values(by=col, ascending=True)
    df_man = df_sorted[df_sorted["method"] == "manual"]
    df_cvx = df_sorted[df_sorted["method"] == "cvxeda"]
    df_nk2 = df_sorted[df_sorted["method"] == "neurokit"]
    fig, ax = plt.subplots(figsize=(10, 6))
    # 's' controls the marker size, 'zorder' ensures points stay on top of the grid lines
    #ax.scatter(df_sorted["segment"], df_sorted[col], color="dodgerblue", edgecolors="black", s=100, zorder=3)
    dot_size = 50
    ax.scatter(df_man["segment"], df_man[col], color="red", edgecolors="black", s=dot_size, zorder=3,  label="manual filtering")
    ax.scatter(df_cvx["segment"], df_cvx[col], color="green", edgecolors="black", s=dot_size, zorder=3, label="CVXEDA")
    ax.scatter(df_nk2["segment"], df_nk2[col], color="blue", edgecolors="black", s=dot_size, zorder=3, label="Neurokit2")


    ax.grid(axis='both', linestyle='--', zorder=1)
    title_name = col.replace('_', ' ').title()
    ax.set_title(f"{title_name} by Segment", fontsize=14, fontweight='bold')
    ax.set_xlabel("Segment", fontsize=12)
    ax.set_ylabel(title_name, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    output = f"{GRAPHS_DIR}/{output_name}.pdf"
    plt.savefig(output, format="pdf")
    #plt.show()

    print(f"Scatter plot saved successfully as {output}")


def analyze_raw():
    df = pd.read_csv("Results/Summary/raw_summary.csv")
    corr = df["max_spearman_mean"][0]
    p_val = df["max_spearman_pval_mean"][0]
    max_corr = df["max_spearman_mean"].idxmax()
    print(f"""
Correlation between Senstream and Biopack is {corr:.2f} and the P value points to a statistically signifficant correlation {p_val} OK
We can see that correlation for the whole measurment is comfortably above 0.7 which means that there is moderate to strong correlation between the 2 sensors""")
    print(f"""Maximum correlation of {df.iloc[max_corr]["max_spearman_mean"]:.2f} has segment {df.iloc[max_corr]["segment"]}
Hypothesis: Making a fist produced a more pronounced eda response. However later analysis proves that this response is of a worse quality
We are displaying normal spearman correlation because max correlation produces much better results for segment matching as it shifts individual segments, thus skewing results
""")
    plot_spearman_scatter(df, "standard_spearman_mean", "raw_spearman_scatter_plot")

def analyze_nk2():
    pass



def analyze_peaks():
    df = pd.read_csv("Results/Summary/peak_summary.csv")
    df_all = df[df["segment"] == "all"].sort_values(by="peak_count_error_mean", ascending=True)
    df_dynamic = df[df["segment"].isin(["dynamic_sitting", "dynamic_standing"])].sort_values(by="peak_count_error_mean", ascending=True)
    df_manual = df[df["method"] == "manual"]
    plot_peak_data(df, "f1_score_mean", "peak_f1_scores")
    plot_peak_data(df, "peak_count_error_mean", "peak_count_errors")
    print(f"""
Although cvxEDA demonstrates a lower absolute peak count error rate (2.53 vs. NeuroKit2's 2.93), NeuroKit2 is selected as the optimal pipeline because it yields a superior F1-score ($0.603$ vs. cvxEDA's $0.563$)
Further analysis also shows that nk2 has a better mae and rmse that is also why we are choosing it
""")

def analyze_summaries():
    df_raw = pd.read_csv("Results/Summary/raw_summary.csv")
    raw_max = df_raw["max_spearman_mean"].idxmax()
    df_raw = df_raw.sort_values(by = "standard_spearman_mean", ascending=False)
    print(df_raw.to_string())
    df_manual = pd.read_csv("Results/Summary/manual_summary.csv")
    df__cvx = pd.read_csv("Results/Summary/cvxeda_summary.csv")
    df_nk2 = pd.read_csv("Results/Summary/nk2_summary.csv")

def filtered_plots(df, columns, output_name):
    """Generates a single multi-panel window containing scatter plots

    comparing manual, cvxEDA, and NeuroKit2 across multiple metrics.
    """
    df = df.sort_values(by=columns)
    df_man = df[df["method"]=="manual"]
    df_cvx = df[df["method"]=="cvxeda"]
    df_nk2 = df[df["method"]=="neurokit"]
    if isinstance(columns, str):
        columns = [columns]

    n_plots = len(columns)

    # 1. Dynamically configure subplot dimensions (1 row, n columns)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6), sharex=True)

    # Ensure axes is always iterable even if only 1 column is passed
    if n_plots == 1:
        axes = [axes]

    dot_size = 100

    # 2. Iterate through each column metric and plot its respective subplot
    for i, col in enumerate(columns):
        ax = axes[i]

        # Plot scatter points (zorder positions them neatly above the gridlines)
        ax.scatter(
            df_man["segment"],
            df_man[col],
            color="red",
            edgecolors="black",
            s=dot_size,
            zorder=3,
            label="Manual Filtering",
            marker="o",
        )
        ax.scatter(
            df_cvx["segment"],
            df_cvx[col],
            color="green",
            edgecolors="black",
            s=dot_size,
            zorder=3,
            label="cvxEDA",
            marker="v",
        )
        ax.scatter(
            df_nk2["segment"],
            df_nk2[col],
            color="blue",
            edgecolors="black",
            s=dot_size,
            zorder=3,
            label="NeuroKit2",
            marker="^",
        )

        # Background grid tracing support
        ax.grid(axis="both", linestyle="--", zorder=1)

        # Dynamic title and label assignments per subplot panels
        title_name = col.replace("_", " ").title()
        ax.set_title(title_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Segment", fontsize=11)
        ax.set_ylabel(title_name, fontsize=11)

        # Rotate labels per axis panel
        ax.tick_params(axis="x", labelrotation=45)

    # 3. Apply a single, unified clean legend on the first panel to avoid duplicating legends
    axes[0].legend(
        loc="upper left", frameon=True, facecolor="white", edgecolor="none"
    )

    plt.tight_layout()
    output = f"{GRAPHS_DIR}/{output_name}.pdf"
    plt.savefig(output, format="pdf")
    #plt.show()

    print(f"Multi-panel plot saved successfully as {output}")

def analyze_filtered():
    df = pd.read_csv("Results/Summary/filtered_summary.csv")
    filtered_plots(df, ["Phasic_MAE_mean", "Tonic_MAE_mean"], "filtered_plots_mae")
    filtered_plots(df, ["Tonic_Spearman_mean","Phasic_Spearman_mean"], "filtered_plots_spearman")
    filtered_plots(df, ["Tonic_Spearman_pval_mean","Phasic_Spearman_pval_mean"], "filtered_plots_pval")
    print("""
cvxEDA is the optimal choice for phasic component analysis when peak detection and SCR dynamics are the primary interest, providing the
highest phasic correlations (ρ = 0.784, p < 10⁻¹³) and best statistical significance. However, for comprehensive analysis of both tonic
and phasic components, NeuroKit2 offers the best balance, with excellent tonic performance (ρ = 0.739) and robust phasic correlations
(ρ = 0.745), while providing the advantage of being a well-established, standardized library in the psychophysiological research
community
""")
    
def get_best_worst_raw_file():
    df_filtered = pd.read_csv("Results/Summary/filtered_summary.csv")

    df_raw = pd.read_csv("Results/raw_correlation_all.csv")
    raw_low_p = df_raw[df_raw["standard_spearman_pval"] == 0]
    raw_representative = raw_low_p["standard_spearman"].idxmax()
    raw_bad_file = df_raw["standard_spearman"].idxmin()

    best_file = raw_low_p.loc[raw_representative]["file_name"]
    worst_file = df_raw.loc[raw_bad_file]["file_name"]

    print(f"best raw file is {best_file} with correlation of {raw_low_p.loc[raw_representative]["standard_spearman"]} and p value of {raw_low_p.loc[raw_representative]["standard_spearman_pval"]}")
    print(f"Worst raw file is {worst_file} with correlation of {df_raw.loc[raw_bad_file]["standard_spearman"]} and p value of {df_raw.loc[raw_bad_file]["standard_spearman_pval"]}")

    return best_file, worst_file


def calculate_raw_spearman_for_file(path):
    """Izračuna surovo Spearmanovo korelacijo za posamezno meritev.

    Datoteke, ki so bile izključene iz skupne analize, nimajo vrstice v
    ``raw_correlation_all.csv``. Za prikaz njihovega grafa pa lahko rho še
    vedno izračunamo neposredno iz izvornih podatkov. Uporabljen je isti
    izsek kot pri skupni analizi: prvih 60 sekund se ne upošteva.
    """
    df = pd.read_csv(path)
    if not {"biopac", "senstream"}.issubset(df.columns):
        return np.nan

    if "time" in df.columns:
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["biopac"] = pd.to_numeric(df["biopac"], errors="coerce")
        df["senstream"] = pd.to_numeric(df["senstream"], errors="coerce")
        df = df.dropna(subset=["time", "biopac", "senstream"])
        if df.empty:
            return np.nan
        elapsed = df["time"] - df["time"].iloc[0]
        df = df.loc[elapsed >= pd.to_timedelta(60, unit="s")]

    clean = df[["biopac", "senstream"]].dropna()
    if len(clean) < 3:
        return np.nan
    correlation, _ = spearmanr(clean["biopac"], clean["senstream"])
    return correlation


def plot_signal_comparison(filename, output_name=None, title=None):
    """Plot one normalized recording and save it using ``output_name``."""
    path = Path(filename)
    if not path.is_file():
        base_dir = REPO_ROOT / "Meritve" / "Combined"
        matches = list(base_dir.rglob(path.name))
        if not matches:
            print(f"Ni datoteke {filename} v {base_dir}")
            return None
        path = matches[0]

    if title is None:
        corr = pd.read_csv(resolve_result_path("raw_correlation_all.csv"))
        row = corr[corr["file_name"].str.casefold() == path.name.casefold()]
        value = row.iloc[0]["standard_spearman"] if len(row) else calculate_raw_spearman_for_file(path)
        if pd.notna(value):
            rho = f"{value:.2f}".replace(".", ",")
            title = rf"Spearmanov $\rho$ = {rho}"
        else:
            title = r"Spearmanov $\rho$ = ni na voljo"

    fig = plot_normalized_signal_comparison(path, title)
    safe_name = output_name or f"signal_{path.stem.replace('_combined', '').replace(' ', '_')}"
    output = THESIS_DIR / f"{safe_name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Graf shranjen: {output}")
    return output

    from scipy.stats import spearmanr
    base_dir = REPO_ROOT / "Meritve" / "Combined"
    matches = list(base_dir.rglob(filename))
    if not matches:
        print(f"Ni datoteke {filename} v {base_dir}")
        return None
    df = pd.read_csv(matches[0])
    

    # Normalize time to seconds
    if 'time' in df.columns:
        try:
            df['time'] = pd.to_datetime(df['time'])
            df['time_sec'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
        except Exception as e:
            print(f"Error normalizing time: {e}")
            df['time_sec'] = range(len(df))

    # Calculate correlation and p-value
    correlation = np.nan
    pvalue = np.nan
    if 'biopac' in df.columns and 'senstream' in df.columns:
        clean = df[['biopac', 'senstream']].dropna()
        if len(clean) >= 3:
            correlation, pvalue = spearmanr(clean['biopac'], clean['senstream'])

    # Define consistent colors for each signal
    COLORS = {
        'biopac': '#1f77b4',  # Blue
        'senstream': '#ff7f0e',  # Orange
    }

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.3)

    # Plot 1: BioPac
    if 'biopac' in df.columns:
        axs[0].plot(df['time_sec'], df['biopac'], color=COLORS['biopac'], linewidth=1.5)
        axs[0].set_title('Biopac', fontsize=12, fontweight='bold')
    else:
        axs[0].set_title('Biopac (Not found)')
    axs[0].set_xlabel('Čas (s)')
    axs[0].set_ylabel('EDA (µS)')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Senstream
    if 'senstream' in df.columns:
        axs[1].plot(df['time_sec'], df['senstream'], color=COLORS['senstream'], linewidth=1.5)
        axs[1].set_title('Senstream', fontsize=12, fontweight='bold')
    else:
        axs[1].set_title('Senstream (Not found)')
    axs[1].set_xlabel('Čas (s)')
    axs[1].set_ylabel('EDA (µS)')
    axs[1].grid(True, linestyle='--', alpha=0.6)


    apply_decimal_comma(fig)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def generate_bland_altman_subplot(ax, b, s, title, unit):
    """Fills a matplotlib axis with a Bland-Altman plot, handling NaNs."""
    # Ensure we are working with clean, aligned data
    mask = ~np.isnan(b) & ~np.isnan(s)
    b = b[mask]
    s = s[mask]

    if len(b) == 0:
        ax.set_title(f"{title} (No Data)")
        return

    # Bland-Altman Math
    mean = (b + s) / 2
    diff = b - s
    md = np.mean(diff)
    sd = np.std(diff)

    # Scatter points
    ax.scatter(mean, diff, alpha=0.5, color='blue', s=1)

    # Statistical lines
    ax.axhline(md, color='red', linestyle=':', linewidth=2, label=f'Bias: {md:.4f}')
    ax.axhline(md + 1.96*sd, color='green', linestyle='--', linewidth=2, label=f'+1.96 SD: {md + 1.96*sd:.4f}')
    ax.axhline(md - 1.96*sd, color='green', linestyle='--', linewidth=2, label=f'-1.96 SD: {md - 1.96*sd:.4f}')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(f'Mean Signal ({unit})', fontsize=10)
    ax.set_ylabel(f'Difference (Biopac - Senstream) ({unit})', fontsize=10)
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_bland_altman(file_path, title="Bland-Altman Analysis", method="manual"):
    """Create Bland-Altman plots for Phasic and Tonic components."""
    # Convert Combined filename to Filtered filename
    filename = Path(file_path).name
    filtered_dir = Path(__file__).parent.parent / "Meritve" / "Filtered" / method

    # Search for the file
    filtered_files = list(filtered_dir.rglob(filename))

    if not filtered_files:
        print(f"Could not find filtered file: {filename} in {filtered_dir}")
        return None

    filtered_path = filtered_files[0]

    try:
        df = pd.read_csv(filtered_path)

        # Check for required columns
        required = ['Biopac_Phasic', 'Senstream_Phasic', 'Biopac_Tonic', 'Senstream_Tonic']
        if not all(col in df.columns for col in required):
            print(f"Missing required columns in {filename}")
            return None

        # Create figure with 2 subplots (side-by-side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Phasic
        b_p = df['Biopac_Phasic'].values
        s_p = df['Senstream_Phasic'].values
        generate_bland_altman_subplot(ax1, b_p, s_p, 'Phasic Component', 'µS')

        # Plot 2: Tonic
        b_t = df['Biopac_Tonic'].values
        s_t = df['Senstream_Tonic'].values
        generate_bland_altman_subplot(ax2, b_t, s_t, 'Tonic Component', 'µS')

        apply_decimal_comma(fig)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


def plot_file_comparison(filename, title=None, output_name=None):
    """Isti graf kot plot_best_worst_files(), a za poljubno meritev.

    filename    ime _combined datoteke, npr. "P012 2_combined.csv"
    title       naslov nad grafom; privzeto Spearmanov rho te meritve, "" za brez naslova
    output_name ime izhodne datoteke brez koncnice; privzeto "signal_<ime meritve>"

    Sliko shrani v thesis/Slike in vrne pot do nje.
    """
    base_dir = REPO_ROOT / "Meritve" / "Combined"
    matches = list(base_dir.rglob(filename))
    if not matches:
        print(f"Ni datoteke {filename} v {base_dir}")
        return None
    if len(matches) > 1:
        # POZOR: rglob na Windowsu ne loci velikosti crk, 'P009 1' (Dan 1) in 'P043 1'
        # (Dan 3) pa sta dva razlicna udelezenca. Zato zahtevamo natancno ujemanje.
        exact = [m for m in matches if m.name == Path(filename).name]
        if len(exact) != 1:
            print(f"Ime {filename} ustreza vec datotekam - podaj pot ali tocno velikost crk:")
            for m in matches:
                print(f"    {m.parent.name}/{m.name}")
            return None
        matches = exact
    path = matches[0]

    stem = Path(filename).stem.replace("_combined", "")
    if title is None:
        corr = pd.read_csv(resolve_result_path("raw_correlation_all.csv"))
        row = corr[corr["file_name"].str.casefold() == Path(filename).name.casefold()]
        value = row.iloc[0]["standard_spearman"] if len(row) else calculate_raw_spearman_for_file(path)
        title = f"Spearmanov $\\rho$ = {value:.2f}".replace(".", ",") if pd.notna(value) else ""

    output_name = output_name or f"signal_{stem.replace(' ', '_')}"
    return plot_signal_comparison(path, output_name=output_name, title=title)



def show_movement_artifact():
    """Primer gibalne motnje v surovem signalu prstana.

    Vse je namenoma zakodirano v funkciji: ena meritev, samo signal Senstream,
    od 60 s naprej (aklimatizacijo izpustimo kot povsod v analizi), z rdece
    osencenima odsekoma, v katerih je motnja vidna.

    Sliko shrani v thesis/Slike, jo prikaze in vrne pot do nje.
    """
    path = REPO_ROOT / "Meritve" / "Combined" / "Dan 1" / "P001 1_combined.csv"
    highlights = [(162, 168), (197, 205)]
    output_name = "primer_surove_gibalne_motnje"

    df = pd.read_csv(path)
    t = pd.to_datetime(df["time"], errors="coerce")
    seconds = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    signal = pd.to_numeric(df["senstream"], errors="coerce").to_numpy()

    keep = seconds >= 120
    seconds, signal = seconds[keep], signal[keep]

    fig, ax = plt.subplots(figsize=(11, 5))
    # oznako da le prvi pas, da se v legendi ne pojavi dvakrat
    for i, (lo, hi) in enumerate(highlights):
        ax.axvspan(lo, hi, color="#f6c3bb", lw=0, zorder=0,
                   label="Gibalna motnja" if i == 0 else None)
    ax.plot(seconds, signal, color=COLOR_SENSTREAM, lw=1.5, zorder=3,
            label="Senstream EDA")

    ax.set_xlabel("Čas (s)", fontsize=11, color=COLOR_INK)
    ax.set_ylabel("Prevodnost kože (µS)", fontsize=11, color=COLOR_INK)
    ax.set_xlim(seconds[0], seconds[-1])
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(color="#e6e6e2", lw=0.6, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_color(COLOR_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9.5)

    apply_decimal_comma(fig)
    plt.tight_layout()
    output = REPO_ROOT / "thesis" / "Slike" / f"{output_name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    print(f"Graf shranjen: {output}")
    #plt.show()
    return output


def show_movement_artifact_filtered():
    """Fazicna in tonicna komponenta istega signala kot show_movement_artifact().

    Isto meritev, isti casovni razpon in ista osencena odseka, le da namesto surovega
    signala prikaze obe komponenti po dekompoziciji z NeuroKit2. Komponenti sta v
    loceni plosci istega okna, ker imata zelo razlicen razpon in bi bila fazicna
    ob tonicni stisnjena v ravno crto.

    Vse je zakodirano v funkciji, slike ne shrani, le prikaze jo.
    """
    path = REPO_ROOT / "Meritve" / "Filtered" / "neurokit2" / "Dan 1" / "P001 1_combined.csv"
    highlights = [(162, 168), (197, 205)]
    output_name = "primer_filtrirane_gibalne_motnje"

    df = pd.read_csv(path)
    t = pd.to_datetime(df["time"], errors="coerce")
    seconds = (t - t.iloc[0]).dt.total_seconds().to_numpy()

    keep = seconds >= 120
    seconds = seconds[keep]

    def column(name):
        return pd.to_numeric(df[name], errors="coerce").to_numpy()[keep]

    # v vsaki plosci obe napravi, da je primerjava neposredna
    panels = [("Tonic", "Tonična komponenta"), ("Phasic", "Fazična komponenta")]
    devices = [("Biopac", "Biopac", COLOR_BIOPAC), ("Senstream", "Senstream", COLOR_SENSTREAM)]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, (suffix, label) in zip(axes, panels):
        # oznako da le prvi pas, da se v legendi ne pojavi dvakrat
        for i, (lo, hi) in enumerate(highlights):
            ax.axvspan(lo, hi, color="#f6c3bb", lw=0, zorder=0,
                       label="Gibalna motnja" if i == 0 else None)
        for prefix, name, color in devices:
            ax.plot(seconds, column(f"{prefix}_{suffix}"), color=color, lw=1.4,
                    zorder=3, label=name)

        ax.set_ylabel(f"{label}\n(SD)", fontsize=10.5, color=COLOR_INK)
        ax.set_xlim(seconds[0], seconds[-1])
        ax.grid(color="#e6e6e2", lw=0.6, zorder=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_color(COLOR_MUTED)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=COLOR_MUTED, labelsize=9.5)

    apply_decimal_comma(fig)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    axes[1].set_xlabel("Čas (s)", fontsize=11, color=COLOR_INK)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    output = REPO_ROOT / "thesis" / "Slike" / f"{output_name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    #plt.show()
    return output


# Pasovi segmentov za sliko razdelitve meritve. Odtenki so svetli, ker so ozadje in
# mora signal ostati berljiv nad njimi; identiteto nosijo napisi, ne barva sama.
# Sosednji pari so preverjeni glede locljivosti pri barvni slepoti (najslabsi ΔE 11,6).
SEGMENT_BANDS = [
    (0, 60, "aklimatizacija", "#e4e3df"),
    (60, 110, "prva naloga", "#aecbe9"),
    (110, 160, "odmor", "#efeee9"),
    (160, None, "druga naloga", "#f5bf99"),
]

# Potemnjena Senstreamova oranzna. Obicajna (#eb6834) ima na svetlih pasovih kontrast
# le 1,9-2,8:1, ta pa najmanj 3,55:1, zato se crta bere cez vse stiri pasove.
COLOR_SENSTREAM_DARK = "#eb6834"




def plot_segment_overview(filename, output_name="razdelitev_segmentov", normalized_comparison=False):
    """Signal z obarvano in oznaceno razdelitvijo meritve na segmente.

    Prikazuje delitev iz razdelka o razdelitvi meritve na sekcije: aklimatizacija
    0-60 s (iz analize izpuscena), 1. naloga 60-110 s, odmor 110-160 s, 2. naloga
    od 160 s do konca. Privzeto prikaze surovi signal Senstream. Ce je
    ``normalized_comparison`` resnicen, prikaze oba signala z enako z-normalizacijo
    kot grafi kategorij korelacij. Sliko shrani v thesis/Slike in vrne pot do nje.
    """
    base_dir = REPO_ROOT / "Meritve" / "Combined"
    matches = list(base_dir.rglob(filename))
    if not matches:
        print(f"Ni datoteke {filename} v {base_dir}")
        return None
    df = pd.read_csv(matches[0])

    t = pd.to_datetime(df["time"], errors="coerce")
    seconds = (t - t.iloc[0]).dt.total_seconds().to_numpy()

    fig, ax = plt.subplots(figsize=(11, 4.6))

    for lo, hi, label, color in SEGMENT_BANDS:
        hi = seconds[-1] if hi is None else hi
        ax.axvspan(lo, hi, color=color, lw=0, zorder=0)
        ax.annotate(label, xy=((lo + hi) / 2, 1.015), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=9.5, color=COLOR_MUTED,
                    linespacing=1.25)

    if normalized_comparison:
        colors = {"biopac": "#1f77b4", "senstream": "#ff7f0e"}
        for column, label in (("biopac", "Biopac"), ("senstream", "Senstream")):
            values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            mean = np.nanmean(values)
            std = np.nanstd(values)
            if not np.isfinite(std) or std == 0:
                raise ValueError(
                    f"Cannot standardize {column} in {filename}: standard deviation is zero or undefined"
                )
            ax.plot(
                seconds,
                (values - mean) / std,
                color=colors[column],
                lw=1.2,
                label=label,
                zorder=3,
            )
        ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.7, zorder=2)
        ax.set_ylabel("Normaliziran signal (SD)", fontsize=11, color=COLOR_INK)
        ax.legend(loc="best", frameon=True, fontsize=9.5)
    else:
        signal = pd.to_numeric(df["senstream"], errors="coerce").to_numpy()
        ax.plot(seconds, signal, color=COLOR_SENSTREAM_DARK, lw=1.5, zorder=3)
        ax.set_ylabel("Prevodnost kože (µS)", fontsize=11, color=COLOR_INK)

    ax.set_xlabel("Čas (s)", fontsize=11, color=COLOR_INK)
    ax.set_xlim(0, seconds[-1])
    ax.grid(color="#ffffff", lw=0.7, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d7d7d2")
    ax.tick_params(colors=COLOR_MUTED, labelsize=9.5)

    apply_decimal_comma(fig)
    plt.tight_layout()
    output = REPO_ROOT / "thesis" / "Slike" / f"{output_name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Graf shranjen: {output}")
    return output


# --- slike za poglavje o filtrirnih metodah ------------------------------------
# Locene od zgornjih angleskih razlicic, ker gredo v nalogo: napisi so slovenski,
# enote pa SD in ne µS, saj so signali pred dekompozicijo z-standardizirani.

METHOD_LABELS = {"manual": "ročno", "neurokit": "NeuroKit2", "cvxeda": "cvxEDA"}
METHOD_DIRS = {"manual": "manual", "neurokit": "neurokit2", "cvxeda": "cvxeda"}


def _bland_altman_axis(ax, b, s, title):
    """Bland-Altmanov graf na eni osi; razlike so v enotah standardnega odklona."""
    mask = ~np.isnan(b) & ~np.isnan(s)
    b, s = b[mask], s[mask]
    if not len(b):
        ax.set_title(f"{title} (ni podatkov)")
        return

    mean, diff = (b + s) / 2, b - s
    md, sd = float(np.mean(diff)), float(np.std(diff))
    # + 0.0 odpravi "-0,000" pri pristranskosti, ki je nicelna po konstrukciji
    num = lambda v: f"{v + 0.0:.3f}".replace("-0.000", "0.000").replace(".", ",")

    ax.scatter(mean, diff, s=3, alpha=0.35, color=COLOR_BIOPAC, lw=0, zorder=3)
    ax.axhline(md, color=COLOR_INK, ls=":", lw=1.4, zorder=4,
               label=f"pristranskost {num(md)}")
    for sign in (1, -1):
        ax.axhline(md + sign * 1.96 * sd, color=COLOR_SENSTREAM, ls="--", lw=1.2, zorder=4,
                   label=f"{'+' if sign > 0 else '−'}1,96 SD = {num(md + sign * 1.96 * sd)}")

    ax.set_xlabel("Povprečje obeh naprav (SD)", fontsize=10, color=COLOR_INK)
    ax.set_ylabel("Razlika Biopac − Senstream (SD)", fontsize=10, color=COLOR_INK)
    # dodatna visina zgoraj, da legenda ne lezi cez zgornjo mejo ujemanja
    ax.margins(y=0.30)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.grid(color="#e6e6e2", lw=0.6, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d7d7d2")
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)


def plot_bland_altman_slo(filename, component="Phasic", methods=("manual", "cvxeda"),
                          output_name=None):
    """Shrani locen Bland-Altmanov graf za vsako izbrano metodo.

    Privzeto primerja fazicno komponento pri rocnem filtriranju in cvxEDA, kar je
    ravno trditev iz besedila: pristranskost je pri obeh nicelna, meje ujemanja pa
    so pri cvxEDA priblizno dvakrat sirse. Skupna os je nujna, sicer bi razlika
    v sirini mej izginila v razlicnem merilu.
    """
    label = {"Phasic": "Fazična", "Tonic": "Tonična"}[component]

    frames = {}
    for method in methods:
        filtered_dir = REPO_ROOT / "Meritve" / "Filtered" / METHOD_DIRS[method]
        matches = list(filtered_dir.rglob(Path(filename).name))
        if not matches:
            print(f"Ni filtrirane datoteke {filename} v {filtered_dir}")
            return None
        frames[method] = pd.read_csv(matches[0])

    output_paths = []
    stem = Path(filename).stem.replace("_combined", "").replace(" ", "_")
    base_name = output_name or f"bland_altman_{stem}"
    method_suffixes = {
        "manual": "rocno",
        "cvxeda": "cvxeda",
    }

    for method in methods:
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        df = frames[method]
        _bland_altman_axis(
            ax,
            df[f"Biopac_{component}"].to_numpy(),
            df[f"Senstream_{component}"].to_numpy(),
            f"{label} komponenta, {METHOD_LABELS[method]}",
        )

        suffix = method_suffixes.get(method, method)
        output = REPO_ROOT / "thesis" / "Slike" / f"{base_name}_{suffix}.pdf"
        output.parent.mkdir(parents=True, exist_ok=True)
        apply_decimal_comma(fig)
        fig.tight_layout()
        fig.savefig(output, format="pdf", bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output)
        print(f"Graf shranjen: {output}")

    return output_paths


def plot_decomposition_slo(filename, output_name="dekompozicija_primerjava"):
    """Tonicna in fazicna komponenta obeh naprav pri vseh treh metodah hkrati.

    Rocno filtriranje in NeuroKit2 se prekrivata, ker izvedeta enak izracun; zato je
    rocna narisana debeleje in NeuroKit2 crtkano cez njo, da je prekrivanje vidno.
    """
    styles = {  # metoda -> (barva, debelina, crta, zorder)
        "manual":   (COLOR_BIOPAC, 2.6, "-", 2),
        "neurokit": (COLOR_INK, 1.1, "--", 3),
        "cvxeda":   (COLOR_SENSTREAM, 1.4, "-", 4),
    }

    frames = {}
    for method, folder in METHOD_DIRS.items():
        matches = list((REPO_ROOT / "Meritve" / "Filtered" / folder).rglob(Path(filename).name))
        if not matches:
            print(f"Ni filtrirane datoteke {filename} za metodo {method}")
            return None
        frames[method] = pd.read_csv(matches[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True)
    panels = [("Biopac_Tonic", "Biopac, tonična"), ("Biopac_Phasic", "Biopac, fazična"),
              ("Senstream_Tonic", "Senstream, tonična"), ("Senstream_Phasic", "Senstream, fazična")]

    for ax, (column, title) in zip(axes.flat, panels):
        for method, (color, lw, ls, z) in styles.items():
            values = frames[method][column].to_numpy()
            ax.plot(np.arange(len(values)) / 10.0, values, color=color, lw=lw, ls=ls,
                    zorder=z, label=METHOD_LABELS[method])
        ax.set_title(title, fontsize=10.5, color=COLOR_INK)
        ax.set_ylabel("Amplituda (SD)", fontsize=9.5, color=COLOR_INK)
        ax.grid(color="#e6e6e2", lw=0.6, zorder=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d7d7d2")
        ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Čas (s)", fontsize=9.5, color=COLOR_INK)
    axes[0][0].legend(loc="upper left", frameon=False, fontsize=9)

    apply_decimal_comma(fig)
    plt.tight_layout()
    output = REPO_ROOT / "thesis" / "Slike" / f"{output_name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Graf shranjen: {output}")
    return output


def average_phasic(column, n_samples=2180):
    """Povprecna fazicna komponenta cez vse meritve za en stolpec.

    Vsako meritev pred sestevanjem standardiziramo, da mocne meritve ne prevladajo.
    Spontani odzivi se pri povprecenju iznicijo, ker se pri vsakem udelezencu zgodijo
    ob drugem casu; odzivi na hiperventilacijo pa se sestejejo, ker so pri vseh ob
    istem casu, saj so vsi gledali isti videoposnetek.
    """
    from file_exclusions import filter_csv_paths

    files = filter_csv_paths(sorted(
        (REPO_ROOT / "Meritve" / "Filtered" / "neurokit2").rglob("*_combined.csv")))
    stack = []
    for path in files:
        v = pd.to_numeric(pd.read_csv(path)[column], errors="coerce").to_numpy()[:n_samples]
        v = np.pad(v, (0, max(0, n_samples - len(v))), constant_values=np.nan)
        stack.append((v - np.nanmean(v)) / np.nanstd(v))
    return np.nanmean(np.vstack(stack), axis=0), len(files)


def plot_average_phasic(output_name="povprecna_fazicna"):
    """Povprecni fazicni komponenti obeh naprav na skupni sliki.

    Skupna slika je namerna: le tako je hkrati vidno, da se vrhova pojavita ob istem
    casu in da je odziv Senstreama manjsi. Izpise tudi case in visine vrhov.
    """
    from scipy.stats import spearmanr

    devices = [("Biopac_Phasic", "Biopac", COLOR_BIOPAC),
               ("Senstream_Phasic", "Senstream", COLOR_SENSTREAM)]

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.axhline(0, color=COLOR_MUTED, lw=0.8, zorder=2)
    curves = {}
    for column, name, color in devices:
        grand, n_files = average_phasic(column)
        curves[name] = grand
        t = np.arange(len(grand)) / 10.0
        ax.plot(t, grand, color=color, lw=1.7, zorder=3, label=name)

        # vrhovi povprecja: lokalni maksimumi nad 1,0 SD, zdruzeni ce so blizu skupaj
        peaks = [i for i in range(1, len(grand) - 1)
                 if grand[i] > 1.0 and grand[i] >= grand[i - 1] and grand[i] > grand[i + 1]]
        merged = []
        for i in peaks:
            if not merged or t[i] - t[merged[-1]] > 10:
                merged.append(i)
            elif grand[i] > grand[merged[-1]]:
                merged[-1] = i
        print(f"{name:10s} vrhovi pri t = "
              + ",  ".join(f"{t[i]:.1f} s ({grand[i]:.2f} SD)" for i in merged))

    # Ujemanje obeh povprecnih krivulj. Prvih 60 s je aklimatizacija in jo tako kot
    # povsod izpustimo. Enota je standardni odklon, ker sta krivulji standardizirani.
    b, s = curves["Biopac"][60 * 10:], curves["Senstream"][60 * 10:]
    mask = ~np.isnan(b) & ~np.isnan(s)
    b, s = b[mask], s[mask]
    rho, pval = spearmanr(b, s)
    print(f"\nUjemanje povprecnih krivulj (od 60 s naprej, n = {len(b)} vzorcev):")
    print(f"  Spearmanov rho = {rho:.3f}  (p = {pval:.2g})")
    print(f"  MAE            = {np.mean(np.abs(b - s)):.3f} SD")
    print(f"  RMSE           = {np.sqrt(np.mean((b - s) ** 2)):.3f} SD\n")

    ax.set_xlabel("Čas (s)", fontsize=11, color=COLOR_INK)
    ax.set_ylabel("Povprečna fazična\nkomponenta (SD)", fontsize=11, color=COLOR_INK)
    ax.set_xlim(0, t[-1])
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(color="#e6e6e2", lw=0.6, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d7d7d2")
    ax.tick_params(colors=COLOR_MUTED, labelsize=9.5)

    apply_decimal_comma(fig)
    plt.tight_layout()
    output = REPO_ROOT / "thesis" / "Slike" / f"{output_name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Graf shranjen: {output}   (povprecje {n_files} meritev)")
    return output


def plot_method_boxplots(output_name="metode_razprsenost"):
    """Porazdelitev metrik po meritvah za vse tri metode.

    Tabele porocajo povprecje +- SD, kar razprsenosti med udelezenci ne pokaze.
    Ta slika jo, in s tem podpre ugotovitev, da so razlike med metodami manjse
    od razlik med posameznimi meritvami.
    """
    quality = {m: pd.read_csv(resolve_result_path(f))
               for m, f in (("manual", "manual_quality_metrics.csv"),
                            ("neurokit", "nk2_quality_metrics.csv"),
                            ("cvxeda", "cvxeda_quality_metrics.csv"))}
    peaks = pd.read_csv(resolve_result_path("peak_detection_all.csv"))

    panels = [
        ("Tonic_Spearman", "Tonična komponenta", "Spearmanov ρ", quality),
        ("Phasic_Spearman", "Fazična komponenta", "Spearmanov ρ", quality),
        ("Phasic_MAE", "Fazična komponenta", "MAE (SD)", quality),
        ("f1_score", "Detekcija vrhov", "F1 mera", None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    for ax, (column, title, ylabel, source) in zip(axes, panels):
        if source is None:
            data = [peaks[peaks["method"] == m][column].dropna() for m in METHOD_LABELS]
        else:
            data = [source[m][column].dropna() for m in METHOD_LABELS]
        bp = ax.boxplot(data, tick_labels=list(METHOD_LABELS.values()), widths=0.55,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=2.5, alpha=0.5,
                                        markerfacecolor=COLOR_MUTED, markeredgecolor="none"),
                        medianprops=dict(color=COLOR_INK, lw=1.6),
                        whiskerprops=dict(color=COLOR_MUTED, lw=1.0),
                        capprops=dict(color=COLOR_MUTED, lw=1.0))
        for patch in bp["boxes"]:
            patch.set(facecolor="#dfe7f0", edgecolor=COLOR_MUTED, lw=1.0)
        ax.set_title(title, fontsize=10.5, color=COLOR_INK)
        ax.set_ylabel(ylabel, fontsize=9.5, color=COLOR_INK)
        ax.grid(axis="y", color="#e6e6e2", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d7d7d2")
        ax.tick_params(colors=COLOR_MUTED, labelsize=9)

    apply_decimal_comma(fig, x=False)
    plt.tight_layout()
    output = REPO_ROOT / "thesis" / "Slike" / f"{output_name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Graf shranjen: {output}")
    return output


def plot_normalized_signal_comparison(file_path, title):
    """Plot one recording's BioPac and Senstream signals after z-score normalization."""
    colors = {
        "biopac": "#1f77b4",
        "senstream": "#ff7f0e",
    }

    fig, ax = plt.subplots(figsize=(15, 5))
    df = pd.read_csv(file_path)

    if "time" in df.columns:
        time = pd.to_datetime(df["time"], errors="coerce")
        if time.notna().all():
            seconds = (time - time.iloc[0]).dt.total_seconds().to_numpy()
        else:
            seconds = np.arange(len(df), dtype=float)
    else:
        seconds = np.arange(len(df), dtype=float)

    for signal in ("biopac", "senstream"):
        if signal not in df.columns:
            continue

        values = pd.to_numeric(df[signal], errors="coerce").to_numpy(dtype=float)
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if not np.isfinite(std) or std == 0:
            raise ValueError(f"Cannot standardize {signal} in {file_path}: standard deviation is zero or undefined")

        normalized = (values - mean) / std
        ax.plot(
            seconds,
            normalized,
            color=colors[signal],
            linewidth=1.2,
            label=signal.capitalize(),
        )

    ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Čas (s)")
    ax.set_ylabel("Normaliziran signal (SD)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    apply_decimal_comma(fig)
    fig.tight_layout()
    return fig


def plot_best_worst_files():
    """Find and plot the best and worst correlated files."""

    best_file, worst_file = get_best_worst_raw_file()

    # Find the full paths
    base_dir = Path(__file__).parent.parent / "Meritve" / "Combined"

    # Search for the files
    best_path = list(base_dir.rglob(best_file))
    worst_path = list(base_dir.rglob(worst_file))

    if not best_path:
        print(f"Could not find best file: {best_file}")
        return
    if not worst_path:
        print(f"Could not find worst file: {worst_file}")
        return

    best_path = best_path[0]
    worst_path = worst_path[0]

    # Plot one normalized figure for each recording.
    print(f"\n=== IDEAL FILE: {best_file} ===")
    best_fig = plot_normalized_signal_comparison(best_path, "Primer idealne meritve")
    best_output = THESIS_DIR / "best_raw_correlation.pdf"
    best_output.parent.mkdir(parents=True, exist_ok=True)
    best_fig.savefig(best_output, format="pdf", bbox_inches="tight")
    plt.close(best_fig)
    print(f"Saved: {best_output}")

    print(f"\n=== BAD FILE: {worst_file} ===")
    worst_fig = plot_normalized_signal_comparison(worst_path, "Primer slabe meritve")
    worst_output = THESIS_DIR / "worst_raw_correlation.pdf"
    worst_output.parent.mkdir(parents=True, exist_ok=True)
    worst_fig.savefig(worst_output, format="pdf", bbox_inches="tight")
    plt.close(worst_fig)
    print(f"Saved: {worst_output}")

    # Plot Bland-Altman for best file
    fig3 = plot_bland_altman(best_path, f"Ideal file example - Bland-Altman", method="manual")
    if fig3:
        fig3.savefig(GRAPHS_DIR / "best_bland_altman.pdf", format="pdf")
        print(f"Saved: {GRAPHS_DIR / 'best_bland_altman.pdf'}")

    # Plot Bland-Altman for worst file
    fig4 = plot_bland_altman(worst_path, f"Bad file example - Bland-Altman", method="manual")
    if fig4:
        fig4.savefig(GRAPHS_DIR / "worst_bland_altman.pdf", format="pdf")
        print(f"Saved: {GRAPHS_DIR / 'worst_bland_altman.pdf'}")

    #plt.show()


def plot_worst_five_files():
    """Plot the five recordings with the lowest raw Spearman correlation.

    The presentation matches ``worst_raw_correlation.pdf``: both complete
    recordings are z-score normalized separately and drawn on the same axis.
    """
    correlation = pd.read_csv(resolve_result_path("raw_correlation_all.csv"))
    worst = correlation.nsmallest(5, "standard_spearman").reset_index(drop=True)
    base_dir = REPO_ROOT / "Meritve" / "Combined"

    fig, axes = plt.subplots(5, 1, figsize=(15, 18))
    colors = {"biopac": "#1f77b4", "senstream": "#ff7f0e"}

    for ax, row in zip(axes, worst.itertuples(index=False)):
        matches = [path for path in base_dir.rglob(row.file_name) if path.name == row.file_name]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one exact match for {row.file_name!r}, found {len(matches)}"
            )

        data = pd.read_csv(matches[0])
        if "time" in data.columns:
            time = pd.to_datetime(data["time"], errors="coerce")
            seconds = ((time - time.iloc[0]).dt.total_seconds().to_numpy()
                       if time.notna().all() else np.arange(len(data), dtype=float))
        else:
            seconds = np.arange(len(data), dtype=float)

        for signal in ("biopac", "senstream"):
            values = pd.to_numeric(data[signal], errors="coerce").to_numpy(dtype=float)
            standard_deviation = np.nanstd(values)
            if not np.isfinite(standard_deviation) or standard_deviation == 0:
                raise ValueError(
                    f"Cannot standardize {signal} in {matches[0]}: "
                    "standard deviation is zero or undefined"
                )
            normalized = (values - np.nanmean(values)) / standard_deviation
            ax.plot(
                seconds,
                normalized,
                color=colors[signal],
                linewidth=1.2,
                label=signal.capitalize(),
            )

        rho = f"{row.standard_spearman:.2f}".replace(".", ",")
        ax.set_title(f"{row.file_name} — Spearmanov $\\rho$ = {rho}", fontsize=11)
        ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.7)
        ax.set_ylabel("Normaliziran signal (SD)")
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Čas (s)")
    apply_decimal_comma(fig)
    fig.tight_layout()

    output = THESIS_DIR / "worst_5_raw_correlations.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Graf shranjen: {output}")
    return output

def plot_decomposed_comparison(filename):
    """
    Load 3 filtered files and create Tonic and Phasic comparison plots for the whole signal.
    All methods for each device/component are plotted on the same graph.

    Parameters:
    - filename: Name of the file (e.g., "P001 1_combined.csv")
    """
    base = Path(__file__).parent.parent / "Meritve" / "Filtered"

    # Find files using rglob
    manual_files = list(base.rglob(f"manual/**/{filename}"))
    neurokit_files = list(base.rglob(f"neurokit2/**/{filename}"))
    cvxeda_files = list(base.rglob(f"cvxeda/**/{filename}"))

    if not manual_files or not neurokit_files or not cvxeda_files:
        print(f"Could not find all files for: {filename}")
        print(f"  Manual: {len(manual_files)} found")
        print(f"  NeuroKit2: {len(neurokit_files)} found")
        print(f"  cvxEDA: {len(cvxeda_files)} found")
        return

    # Load data
    df_manual = pd.read_csv(manual_files[0])
    df_manual['time'] = pd.to_datetime(df_manual['time'], errors='coerce')

    df_nk = pd.read_csv(neurokit_files[0])
    df_nk['time'] = pd.to_datetime(df_nk['time'], errors='coerce')

    df_cvx = pd.read_csv(cvxeda_files[0])
    df_cvx['time'] = pd.to_datetime(df_cvx['time'], errors='coerce')

    data = {
        'manual': df_manual,
        'neurokit2': df_nk,
        'cvxeda': df_cvx
    }

    # Method colors, labels, and line widths
    method_colors = {
        'manual': 'tab:red',
        'neurokit2': 'tab:blue',
        'cvxeda': 'tab:green'
    }
    method_labels = {
        'manual': 'Manual',
        'neurokit2': 'NeuroKit2',
        'cvxeda': 'cvxEDA'
    }
    method_linewidths = {
        'manual': 2.0,  # Thicker to show through neurokit2
        'neurokit2': 1.2,
        'cvxeda': 1.2
    }

    # Create figure: 2 rows (BioPac, Senstream) x 2 cols (Tonic, Phasic)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle(f'Decomposed Signal Comparison - {filename}', fontsize=16, fontweight='bold')

    # Plot each component
    for col_idx, component in enumerate(['Tonic', 'Phasic']):
        # BioPac (top row)
        ax_bio = axes[0, col_idx]
        for method in ['manual', 'neurokit2', 'cvxeda']:
            df = data[method]
            elapsed = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
            ax_bio.plot(elapsed, df[f'Biopac_{component}'],
                       color=method_colors[method],
                       linewidth=method_linewidths[method],
                       label=method_labels[method])

        ax_bio.set_title(f'Biopac {component}', fontweight='bold', fontsize=12)
        ax_bio.set_ylabel('EDA (μS)', fontsize=10)
        ax_bio.grid(True, alpha=0.3)
        ax_bio.legend(loc='best', fontsize=9)

        # Senstream (bottom row)
        ax_sen = axes[1, col_idx]
        for method in ['manual', 'neurokit2', 'cvxeda']:
            df = data[method]
            elapsed = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
            ax_sen.plot(elapsed, df[f'Senstream_{component}'],
                       color=method_colors[method],
                       linewidth=method_linewidths[method],
                       label=method_labels[method])

        ax_sen.set_title(f'Senstream {component}', fontweight='bold', fontsize=12)
        ax_sen.set_ylabel('EDA (μS)', fontsize=10)
        ax_sen.set_xlabel('Čas (s)', fontsize=10)
        ax_sen.grid(True, alpha=0.3)
        ax_sen.legend(loc='best', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save
    output_dir = GRAPHS_DIR / "decomposed_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = filename.replace('.csv', '').replace(' ', '_')

    output_path = output_dir / f'{safe_name}_decomposed_comparison.pdf'
    fig.savefig(output_path, format='pdf', bbox_inches='tight')

    output_path_png = output_dir / f'{safe_name}_decomposed_comparison.png'
    fig.savefig(output_path_png, format='png', dpi=150, bbox_inches='tight')

    print(f'Saved decomposed comparison to: {output_path} and {output_path_png}')
    plt.close(fig)


# Task onsets in the video protocol, seconds from recording start (same timeline as
# correlation_by_parts.py: 0-60 rest, 60-100 task 1, 100-160 break, 160-end task 2).
TASK_ONSETS = {
    60: "Primer stoje mirovanje",
    160: "Primer sede premikanje",
}
ANTICIPATION_WINDOW = 10  # seconds before a task onset, highlighted as anticipation
SLICE_LENGTH = 40  # length of the plotted slice, centered on the task onset

# One window per entry: (file, onset of the task of interest)
ANTICIPATION_FILES = [
    ("Meritve/Combined/Dan 1/P002 2_combined.csv", 60),
    ("Meritve/Combined/Dan 1/P008 1_combined.csv", 160),
]


def plot_task_anticipation(anticipation_window=ANTICIPATION_WINDOW, slice_length=SLICE_LENGTH):
    """
    Function plots some files where respondants were anticipating the task producing an EDA response
    """

    for file, onset in ANTICIPATION_FILES:
        df = pd.read_csv(REPO_ROOT / file)
        time = pd.to_datetime(df['time'])
        df['elapsed'] = (time - time.iloc[0]).dt.total_seconds()

        # slice centered on the task onset, so the pre-task rise is visible
        start, end = onset - slice_length / 2, onset + slice_length / 2
        window = df[df['elapsed'].between(start, end)]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(window['elapsed'], window['senstream'], color='#ff7f0e', linewidth=1.5,
                label='Senstream EDA')
        ax.axvline(onset, color='black', linestyle='--', linewidth=1.2, label='Pričetek naloge')
        ax.axvspan(onset - anticipation_window, onset, color='tab:red', alpha=0.12,
                   label='Pričakovanje začetka')

        ax.set_xlim(start, end)
        ax.set_xlabel('Čas (s)', fontsize=10)
        ax.set_ylabel('EDA (µS)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', fontsize=9)
        apply_decimal_comma(fig)
        fig.tight_layout()
        output = f"thesis/Slike/{TASK_ONSETS[onset]}.pdf"
        plt.savefig(output,format="pdf")

    #plt.show()


# --- gibanje iz pospeskometra --------------------------------------------------
# Jakost gibanja po segmentih izracuna Scripts/movement_metrics.py.

# Zacetek 2. naloge (enako kot BREAK_TIME_2_END v correlation_by_parts).
TASK2_START_S = 160

# Rezervni fiksni prag, ce prilagodljivega ni mogoce izracunati. Enotnega praga za obe
# meritvi ni: stiskanje pesti sede da povprecno 0,22 g, premikanje roke stoje pa 0,63 g
# (Results/Summary/movement_summary.csv), zato fiksna vrednost zgresi enega ali drugega.
MOVEMENT_THRESHOLD_G = 0.2


# Kategoricni barvi za dve napravi (preverjeno: CVD dE 24,7; kontrast > 3:1).
# Barvi nosita identiteto naprave, senceno obmocje je nevtralno, ker je opomba in ne serija.
COLOR_BIOPAC = "#2a78d6"
COLOR_SENSTREAM = "#eb6834"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_SHADE = "#c9c9c4"


def load_accel(path):
    """Vrne (sekunde od zacetka, norma pospeska) iz _combined datoteke."""
    df = pd.read_csv(path)
    t = pd.to_datetime(df["time"], errors="coerce")
    seconds = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    accel = pd.to_numeric(df["net_acceleration"], errors="coerce").to_numpy()
    return seconds, accel


def adaptive_threshold(seconds, accel):
    """Prag, prilagojen posamezni meritvi: povprecni pospesek v 2. nalogi.

    Fiksni prag ne deluje, ker se med meritvami mocno razlikuje jakost gibanja, ne pa
    mirovanje: mediana mirovanja je 0,029 g (IQR 0,022-0,036), gibanje v 2. nalogi pa
    ima mediano 1,52 g (IQR 0,49-2,57), sede petkrat manj kot stoje. Prag mora zato
    slediti lastnemu razponu meritve in ne sumnemu dnu.

    Povprecje 2. naloge to naredi samo od sebe: 2. naloga je mesanica dveh sunkov in
    mirovanja med njima, zato povprecje pade nekam vmes - priblizno na tretjino jakosti
    sunka, odvisno od tega, kako dolgo je oseba dejansko premikala roko. Metoda nima
    nastavljive konstante.

    Omejitev: prag se racuna iz segmenta, v katerem gibanje tudi iscemo, in predpostavlja
    znan merilni protokol. Za prikaz in izlocanje odsekov je to ustrezno, za oceno
    uspesnosti zaznavanja pa ne, prav tako ne za meritve v naravnem okolju brez nalog.
    """
    values = accel[seconds >= TASK2_START_S]
    values = values[~np.isnan(values)]
    if not len(values):
        return MOVEMENT_THRESHOLD_G
    return float(values.mean())


def movement_windows(seconds, accel, threshold=None, window_seconds=10):
    """Okna, v katerih povprecni pospesek preseze prag. Vrne seznam (zacetek, konec).

    Ce prag ni podan, ga za to meritev doloci adaptive_threshold().
    """
    if threshold is None:
        threshold = adaptive_threshold(seconds, accel)
    edges = np.arange(0, seconds[-1] + window_seconds, window_seconds)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (seconds >= lo) & (seconds < hi)
        if m.sum() and np.nanmean(accel[m]) > threshold:
            out.append((lo, hi))
    return out


def merge_adjacent(windows):
    """Sosednja okna zdruzi v eno obmocje gibanja - dve sosednji okni sta en sunek."""
    regions = []
    for lo, hi in windows:
        if regions and regions[-1][1] == lo:
            regions[-1] = (regions[-1][0], hi)
        else:
            regions.append((lo, hi))
    return regions


def plot_movement_highlight(filename=None, window_seconds=10, output_name=None,
                            threshold=None):
    """Surova signala z osencenimi odseki, v katerih je gibanje preseglo prag.

    Signala sta z-standardizirana, ker imata napravi razlicni merilni obmocji;
    graf z dvema navpicnima osema bi razmerje med njima popacil.
    Prag se privzeto doloci za vsako meritev posebej (adaptive_threshold).
    """
    combined_dir = REPO_ROOT / "Meritve" / "Combined"

    if filename is None:
        # reprezentativna meritev stoje: mediana korelacije, da graf ni izbran po ucinku
        corr = pd.read_csv(resolve_result_path("raw_correlation_standing.csv"))
        corr = corr.dropna(subset=["standard_spearman"]).sort_values("standard_spearman")
        filename = corr.iloc[len(corr) // 2]["file_name"]
        print(f"Izbrana reprezentativna meritev stoje (mediana korelacije): {filename}")

    matches = list(combined_dir.rglob(filename))
    if not matches:
        print(f"Ni datoteke {filename} v {combined_dir}")
        return None
    df = pd.read_csv(matches[0])

    t = pd.to_datetime(df["time"], errors="coerce")
    seconds = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    accel = pd.to_numeric(df["net_acceleration"], errors="coerce").to_numpy()

    def z(col):
        v = pd.to_numeric(df[col], errors="coerce").to_numpy()
        return (v - np.nanmean(v)) / np.nanstd(v)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1]})

    # osenceni odseki: 10-sekundna okna, v katerih povprecni pospesek preseze prag
    if threshold is None:
        threshold = adaptive_threshold(seconds, accel)
    shaded = movement_windows(seconds, accel, threshold, window_seconds)
    for ax in axes:
        for lo, hi in shaded:
            ax.axvspan(lo, hi, color=COLOR_SHADE, alpha=0.55, lw=0, zorder=0)

    axes[0].plot(seconds, z("biopac"), color=COLOR_BIOPAC, lw=1.4, label="Biopac", zorder=4)
    axes[0].plot(seconds, z("senstream"), color=COLOR_SENSTREAM, lw=1.4, label="Senstream", zorder=3)
    axes[0].set_ylabel("Normalizirana\nprevodnost", fontsize=11, color=COLOR_INK)
    axes[0].legend(loc="upper left", frameon=False, fontsize=10)

    axes[1].plot(seconds, accel, color=COLOR_MUTED, lw=1.0, zorder=3)
    # logaritemska os: vrhovi dosezejo nekaj g, prag pa je pri 0,2 g -
    # pri linearni osi bi prag lezal na ničli in prehodov ne bi bilo mogoce brati
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.01, max(1.0, float(np.nanmax(accel)) * 1.6))
    axes[1].set_ylabel("Pospešek (g)", fontsize=11, color=COLOR_INK)
    axes[1].set_xlabel("Čas (s)", fontsize=11, color=COLOR_INK)

    for ax in axes:
        ax.grid(color="#e6e6e2", lw=0.6, zorder=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d7d7d2")
        ax.tick_params(colors=COLOR_MUTED, labelsize=9.5)
    axes[0].set_xlim(0, seconds[-1])

    plt.tight_layout()
    safe = Path(filename).stem.replace(" ", "_")
    output_name = output_name or f"gibanje_{safe}"
    output = REPO_ROOT / "thesis" / "Slike" / f"{output_name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Graf shranjen: {output}   ({len(shaded)} osencenih oken po {window_seconds} s)")
    return output


def count_movement_regions(threshold=None, window_seconds=10):
    """Presteje zaznana obmocja gibanja v vseh meritvah.

    Uporablja isto logiko kot plot_movement_highlight, le da sosednja okna zdruzi
    v eno obmocje. Po protokolu sta v 2. nalogi dva sunka gibanja, zato pri vsaki
    meritvi pricakujemo dve obmocji. Izpise porazdelitev in nasteje odstopanja.

    Prag se privzeto doloci za vsako meritev posebej; s `threshold` ga fiksiras.
    Vrne DataFrame z eno vrstico na meritev.
    """
    from file_exclusions import filter_csv_paths

    files = filter_csv_paths(sorted((REPO_ROOT / "Meritve" / "Combined").rglob("*_combined.csv")))

    rows = []
    for path in files:
        seconds, accel = load_accel(path)
        if np.isnan(accel).all():
            print(f"  BREZ POSPESKA: {path.name}")
            continue
        thr = adaptive_threshold(seconds, accel) if threshold is None else threshold
        regions = merge_adjacent(movement_windows(seconds, accel, thr, window_seconds))
        rows.append({
            "file": path.name,
            # sede = meritev 1, stoje = meritev 2 (enako kot drugod v analizi)
            "condition": "stoje" if path.stem.endswith("2_combined") else "sede",
            "threshold_g": thr,
            "n_regions": len(regions),
            # obmocja pred zacetkom 2. naloge protokol ne predvideva
            "n_in_task2": sum(1 for lo, _ in regions if lo >= TASK2_START_S),
            "regions": " ".join(f"{int(lo)}-{int(hi)}" for lo, hi in regions),
        })

    df = pd.DataFrame(rows)
    kind = (f"fiksni prag {threshold} g" if threshold is not None
            else f"prilagodljiv prag {df.threshold_g.min():.2f}-{df.threshold_g.max():.2f} g "
                 f"(mediana {df.threshold_g.median():.2f})")
    print(f"\n{kind}, okno {window_seconds} s   (N = {len(df)} meritev)")

    print("\nPorazdelitev stevila obmocij:")
    for n, k in df.n_regions.value_counts().sort_index().items():
        mark = "   <-- pricakovano" if n == 2 else ""
        print(f"  {n} obmocij: {k:3d} meritev{mark}")

    for label, sel in (("natanko 2", df.n_regions == 2),
                       ("manj kot 2", df.n_regions < 2),
                       ("vec kot 2", df.n_regions > 2)):
        print(f"  {label:11s}: {int(sel.sum()):3d} ({sel.mean():.0%})")

    print("\nPo pogoju (delez z natanko 2 obmocjema):")
    for cond, group in df.groupby("condition"):
        print(f"  {cond:6s}: {int((group.n_regions == 2).sum()):3d}/{len(group)} "
              f"({(group.n_regions == 2).mean():.0%})")

    clean = (df.n_regions == 2) & (df.n_in_task2 == 2)
    print(f"\nNatanko dve obmocji in obe v 2. nalogi: {int(clean.sum())}/{len(df)} ({clean.mean():.0%})")

    odd = df[df.n_regions != 2]
    if len(odd):
        print(f"\nMeritve, ki odstopajo od dveh ({len(odd)}):")
        for r in odd.itertuples():
            print(f"  {r.file:32s} {r.condition:6s} prag {r.threshold_g:5.3f}  "
                  f"{r.n_regions} obmocij ({r.n_in_task2} v 2. nalogi)   {r.regions}")

    return df

def plot_corr_categories():
    plot_signal_comparison("P028 2_combined.csv", "kategorija_zelo_dobri")
    plot_signal_comparison("P018 1_combined.csv", "kategorija_dobri")
    plot_signal_comparison("P033 2_combined.csv", "kategorija_sprejemljivi")
    plot_signal_comparison("P036 1_combined.csv", "kategorija_slabi")

if __name__ == "__main__":
    #os.makedirs(RESULTS_DIR, exist_ok=True)
    #os.makedirs(GRAPHS_DIR, exist_ok=True)
    #make_box_plots(["Tonic_MAE", "Phasic_MAE" ,"Tonic_RMSE", "Phasic_RMSE", "Tonic_Spearman", "Phasic_Spearman"])
    #analyze_raw()
    #analyze_filtered()
    #analyze_peaks()
    #get_best_worst_raw_file()
    #plot_best_worst_files()
    #plot_decomposed_comparison("P028 2_combined.csv")
    #plot_task_anticipation()
    #count_movement_regions()
    # meritev je izbrana rocno, ker je slika v nalogi vezana na to ime datoteke
    #plot_movement_highlight("P002 2_combined.csv")
    #plot_average_phasic()
    #show_movement_artifact()
    #show_movement_artifact_filtered()
    #plot_file_comparison("P019 1_combined.csv")
    #plot_segment_overview("P028 2_combined.csv")
    #plot_corr_categories()
    #plot_segment_overview("P028 2_combined.csv", output_name="2_odziv_vecji", normalized_comparison=True)
    #plot_movement_highlight("P023 1_combined.csv", output_name="gibanje_extra_detekcija")
    plot_bland_altman_slo("P028 2_combined.csv")
