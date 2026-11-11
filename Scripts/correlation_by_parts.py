import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from file_exclusions import filter_csv_paths

PRINT_FILE_ROWS = False

CATEGORIES = [
    ("Great (>= 0.9)", 0.9),
    ("Strong (>= 0.7)", 0.7),
    ("Medium (0.3 - 0.7)", 0.3),
]
SPLIT_TIME = pd.to_timedelta(109, unit="s") #this is the split for sitting and standing
BREAK_TIME_1_START = pd.to_timedelta(0, unit="s")
BREAK_TIME_1_END = pd.to_timedelta(60, unit="s")
BREAK_TIME_2_START = pd.to_timedelta(110, unit="s")
BREAK_TIME_2_END = pd.to_timedelta(160, unit="s")
#Video segments 0-60 mirovanje, 60-100 hiperventilacija static 100-160 mirovanje, 160-x hiperventilacija dynamic

def bland_altman_metrics(x, y):
    diff = x - y
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    loa_lower = mean_diff - 1.96 * sd_diff
    loa_upper = mean_diff + 1.96 * sd_diff
    return mean_diff, loa_lower, loa_upper

def error_metrics(x, y):
    error = x - y
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    
    x_range = np.max(x) - np.min(x)
    nrmse_range = rmse / x_range if x_range != 0 else np.nan
    
    x_std = np.std(x, ddof=1)
    nrmse_std = rmse / x_std if x_std != 0 else np.nan
    
    return mae, rmse, nrmse_range, nrmse_std

def categorize(value: float) -> str:
    if value >= 0.9:
        return "Great"
    if value >= 0.7:
        return "Strong"
    if value >= 0.3:
        return "Medium"
    return "Weak"


def init_category_buckets() -> dict:
    return {
        "Great (>= 0.9)": [],
        "Strong (>= 0.7)": [],
        "Medium (0.3 - 0.7)": [],
        "Weak (< 0.3)": [],
    }


def compute_spearman(df: pd.DataFrame) -> tuple[float, float]:
    """Compute Spearman correlation and p-value."""
    clean = df[["biopac", "senstream"]].dropna()
    return spearmanr(clean["biopac"], clean["senstream"])


def compute_max_cross_corr(df: pd.DataFrame, max_shift: int = 100) -> tuple[float, int, float]:
    """Compute max cross-correlation with lag and p-value."""
    best_corr = -1.0
    best_lag = 0
    best_pval = np.nan

    for shift in range(-max_shift, max_shift + 1):
        biopac_series = df["biopac"]
        senstream_shifted = df["senstream"].shift(shift)
        mask = ~(biopac_series.isna() | senstream_shifted.isna())
        x_clean = biopac_series[mask].to_numpy()
        y_clean = senstream_shifted[mask].to_numpy()

        if len(x_clean) > 0:
            corr, pval = spearmanr(x_clean, y_clean)
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_lag = shift
                best_pval = pval

    if best_corr == -1.0:
        return np.nan, 0, np.nan

    return best_corr, best_lag, best_pval


def print_metric_summary(title: str, results: list[float], categories: dict, lags: list[int] | None = None) -> None:
    if not results:
        print(f"\n{title}: No valid data")
        return

    print(f"\n{title}")
    print("=" * len(title))
    print(f"Total Files Analyzed: {len(results)}")
    print(f"Average:              {np.mean(results):.4f}")
    print(f"Maximum:              {np.max(results):.4f}")
    print(f"Minimum:              {np.min(results):.4f}")

    if lags is not None:
        print(f"Avg Absolute Lag:     {np.mean(np.abs(lags)) * 0.1:.2f}s")

    print("Counts By Category:")
    for key, _ in CATEGORIES:
        print(f"- {key:<18}: {len(categories[key])}")
    print(f"- {'Weak (< 0.3)':<18}: {len(categories['Weak (< 0.3)'])}")


def correlation_by_parts(file_paths: list[Path], results_dir: Path, output_prefix: str = "") -> None:
    if not file_paths:
        print("No files provided for correlation_by_parts")
        return

    name_mapping = {
        "static": "raw_correlation_static.csv",
        "dynamic": "raw_correlation_dynamic.csv",
        "break": "raw_correlation_break.csv",
    }

    segment_state = {
        "static": {
            "spearman": [],
            "spearman_categories": init_category_buckets(),
            "cross": [],
            "cross_categories": init_category_buckets(),
            "lags": [],
        },
        "dynamic": {
            "spearman": [],
            "spearman_categories": init_category_buckets(),
            "cross": [],
            "cross_categories": init_category_buckets(),
            "lags": [],
        },
        "break": {
            "spearman": [],
            "spearman_categories": init_category_buckets(),
            "cross": [],
            "cross_categories": init_category_buckets(),
            "lags": [],
        },
    }

    segment_records = {
        "static": [],
        "dynamic": [],
        "break": [],
    }

    if PRINT_FILE_ROWS:
        print(
            f"{'File Name':<45} | {'Part':<8} | {'Spearman':<10} | {'Max Corr':<10} | {'Lag (s)':<8} | {'Category':<10}"
        )
        print("-" * 108)

    for file_path in file_paths:
        file_path = Path(file_path)
        
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            print(f"Error reading {file_path.name}: {exc}")
            continue

        required = {"time", "biopac", "senstream"}
        if not required.issubset(df.columns):
            print(f"{file_path.name:<45} | {'N/A':<8} | {'Missing Cols':<10} | {'N/A':<10} | {'N/A':<8} | {'N/A':<10}")
            continue

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["biopac"] = pd.to_numeric(df["biopac"], errors="coerce")
        df["senstream"] = pd.to_numeric(df["senstream"], errors="coerce")
        df = df.dropna(subset=["time", "biopac", "senstream"])

        elapsed = df["time"] - df["time"].iloc[0]
        break_mask = ((elapsed > BREAK_TIME_2_START) & (elapsed < BREAK_TIME_2_END))
        static_mask = (elapsed > BREAK_TIME_1_END) & (elapsed <= BREAK_TIME_2_START)
        dynamic_mask = elapsed > BREAK_TIME_2_END
        segments = {
            "static": df[static_mask],
            "dynamic": df[dynamic_mask],
            "break": df[break_mask],
        }

        for segment_name, segment_df in segments.items():
            if segment_df.empty:
                if PRINT_FILE_ROWS:
                    print(f"{file_path.name:<45} | {segment_name:<8} | {'Empty':<10} | {'Empty':<10} | {'N/A':<8} | {'N/A':<10}")
                continue

            spearman, spearman_pval = compute_spearman(segment_df)
            max_corr, lag, max_pval = compute_max_cross_corr(segment_df, max_shift=100)

            if np.isnan(spearman) or np.isnan(max_corr):
                if PRINT_FILE_ROWS:
                    print(f"{file_path.name:<45} | {segment_name:<8} | {'NaN':<10} | {'NaN':<10} | {'N/A':<8} | {'N/A':<10}")
                continue

            category = categorize(max_corr)
            lag_sec = lag * 0.1

            clean_seg = segment_df[["biopac", "senstream"]].dropna()
            x = clean_seg["biopac"].to_numpy()
            y = clean_seg["senstream"].to_numpy()

            mae, rmse, nrmse_range, nrmse_std = error_metrics(x, y)
            if PRINT_FILE_ROWS:
                print(
                    f"{file_path.name:<45} | {segment_name:<8} | {spearman:>10.4f} | {max_corr:>10.4f} | {lag_sec:>7.1f}s | {category:<10}"
                )

            state = segment_state[segment_name]

            state["spearman"].append(spearman)
            if spearman >= 0.9:
                state["spearman_categories"]["Great (>= 0.9)"].append(file_path.name)
            elif spearman >= 0.7:
                state["spearman_categories"]["Strong (>= 0.7)"].append(file_path.name)
            elif spearman >= 0.3:
                state["spearman_categories"]["Medium (0.3 - 0.7)"].append(file_path.name)
            else:
                state["spearman_categories"]["Weak (< 0.3)"].append(file_path.name)

            state["cross"].append(max_corr)
            state["lags"].append(lag)
            if max_corr >= 0.9:
                state["cross_categories"]["Great (>= 0.9)"].append(file_path.name)
            elif max_corr >= 0.7:
                state["cross_categories"]["Strong (>= 0.7)"].append(file_path.name)
            elif max_corr >= 0.3:
                state["cross_categories"]["Medium (0.3 - 0.7)"].append(file_path.name)
            else:
                state["cross_categories"]["Weak (< 0.3)"].append(file_path.name)

            segment_records[segment_name].append({
                "file_name": file_path.name,
                "standard_spearman": spearman,
                "standard_spearman_pval": spearman_pval,
                "max_spearman": max_corr,
                "max_spearman_pval": max_pval,
                "lag_seconds": lag_sec,
                "raw_nrmse_range": nrmse_range,
                "raw_nrmse_std": nrmse_std,
            })

    for segment_name, state in segment_state.items():
        heading = f"{segment_name.upper()} - STANDARD SPEARMAN SUMMARY"
        print_metric_summary(heading, state["spearman"], state["spearman_categories"])

        heading = f"{segment_name.upper()} - MAX CROSS-CORRELATION SUMMARY"
        print_metric_summary(heading, state["cross"], state["cross_categories"], lags=state["lags"])

    for segment_name, rows in segment_records.items():
        csv_name = f"{output_prefix}{name_mapping[segment_name]}"
        csv_path = results_dir / csv_name
        df_out = pd.DataFrame(rows)
        df_out.to_csv(csv_path, index=False)
        print(f"\n Saved {segment_name} segment results to {csv_path}")


def main():
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent / "Meritve" / "Combined"
    results_dir = script_path.parent.parent / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        raise SystemExit(1)

    csv_files = filter_csv_paths(sorted(base_dir.rglob("*_combined.csv")))
    csv_sitting = filter_csv_paths(sorted(base_dir.rglob("*1_combined.csv")))
    csv_standing = filter_csv_paths(sorted(base_dir.rglob("*2_combined.csv")))
    if not csv_files:
        print(f"No combined CSV files found in {base_dir}")
        raise SystemExit(1)

    correlation_by_parts(csv_files, results_dir)
    if csv_sitting:
        correlation_by_parts(csv_sitting, results_dir, output_prefix="sitting_")
    if csv_standing:
        correlation_by_parts(csv_standing, results_dir, output_prefix="standing_")


if __name__ == "__main__":
    main()
