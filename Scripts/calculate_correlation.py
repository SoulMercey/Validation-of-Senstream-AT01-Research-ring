import os
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from file_exclusions import filter_csv_paths, is_excluded

PRINT_FILE_ROWS = False


def filter_out_initial_60_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the first 60 seconds of measurements when a time column is available. This is beacuse we are treating this as baseline and not analyzing it"""
    if "time" not in df.columns:
        return df

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["biopac"] = pd.to_numeric(df["biopac"], errors="coerce")
    df["senstream"] = pd.to_numeric(df["senstream"], errors="coerce")
    df = df.dropna(subset=["time", "biopac", "senstream"])

    if df.empty:
        return df

    elapsed = df["time"] - df["time"].iloc[0]
    return df.loc[elapsed >= pd.to_timedelta(60, unit="s")].copy()


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

### We cant use pearsons correlation as our data is not normally distributed. Therefore we use spearmans
def calculate_correlations(file_list):
    results = []
    p_values = []
    file_corrs = {}
    categories = {
        'Great (>= 0.9)': [],
        'Strong (>= 0.7)': [],
        'Medium (0.3 - 0.7)': [],
        'Weak (< 0.3)': []
    }
    
    if PRINT_FILE_ROWS:
        print(f"{'File Name':<45} | {'Correlation':<12} | {'Category':<10}")
        print("-" * 75)
    
    for file_path in file_list:
        if is_excluded(file_path):
            continue
        try:
            df = pd.read_csv(file_path)
            
            if 'biopac' in df.columns and 'senstream' in df.columns:
                df = filter_out_initial_60_seconds(df)
                if df.empty:
                    if PRINT_FILE_ROWS:
                        print(f"{file_path.name:<45} | {'Empty After Trim':>12} | {'N/A':<10}")
                    continue

                clean = df[['biopac', 'senstream']].dropna()
                if clean.empty:
                    if PRINT_FILE_ROWS:
                        print(f"{file_path.name:<45} | {'No Valid Data':>12} | {'N/A':<10}")
                    continue

                x = clean['biopac'].to_numpy()
                y = clean['senstream'].to_numpy()
                
                correlation, p_value = spearmanr(x, y)
                
                if not np.isnan(correlation):
                    results.append(correlation)
                    
                    mae, rmse, nrmse_range, nrmse_std = error_metrics(x, y)                    
                    file_corrs[file_path.name] = {
                        "standard_spearman": correlation,
                        "p_value": p_value,
                        "nrmse_range": nrmse_range,
                        "nrmse_std": nrmse_std,
                    }
                    
                    # Categorize
                    if correlation >= 0.9:
                        cat = 'Great'
                        categories['Great (>= 0.9)'].append(file_path.name)
                    elif correlation >= 0.7:
                        cat = 'Strong'
                        categories['Strong (>= 0.7)'].append(file_path.name)
                    elif correlation >= 0.3:
                        cat = 'Medium'
                        categories['Medium (0.3 - 0.7)'].append(file_path.name)
                    else:
                        cat = 'Weak'
                        categories['Weak (< 0.3)'].append(file_path.name)
                        #results.pop()  # Remove weak correlation from results for summary stats
                        
                    if PRINT_FILE_ROWS:
                        print(f"{file_path.name:<45} | {correlation:>12.4f} | {cat:<10}")
                else:
                    if PRINT_FILE_ROWS:
                        print(f"{file_path.name:<45} | {'NaN (No Var)':>12} | {'N/A':<10}")
            else:
                if PRINT_FILE_ROWS:
                    print(f"{file_path.name:<45} | {'Missing Cols':>12} | {'N/A':<10}")
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    if results:
        avg_corr = np.mean(results)
        max_corr = np.max(results)
        min_corr = np.min(results)
        
        print("\n" + "="*35)
        print("📊 CORRELATION SUMMARY")
        print("="*35)
        print(f"Total Files Analyzed: {len(results)}")
        print(f"Average Correlation:  {avg_corr:.4f}")
        print(f"Maximum Correlation:  {max_corr:.4f}")
        print(f"Minimum Correlation:  {min_corr:.4f}")
        print("-" * 35)
        
        print("COUNTS BY CATEGORY:")
        for cat_name, files in categories.items():
            print(f"- {cat_name:<18}: {len(files)}")
            
        if categories['Weak (< 0.3)']:
            print("\n❌ WEAK CORRELATIONS (LIST):")
            for name in categories['Weak (< 0.3)']:
                print(f"  - {name}")
        
        print("="*35)
    else:
        print("\nNo valid correlation data found.")
    return file_corrs

def calculate_max_cross_correlations(file_list, max_shift=100):
    """Finds the maximum correlation by shifting Senstream data against BioPack."""
    results = []
    lags = []
    file_max_corrs = {}
    categories = {
        'Great (>= 0.9)': [],
        'Strong (>= 0.7)': [],
        'Medium (0.3 - 0.7)': [],
        'Weak (< 0.3)': []
    }
    
    if PRINT_FILE_ROWS:
        print(f"\n{'File Name':<45} | {'Max Corr':<10} | {'Lag (s)':<8} | {'Category':<10}")
        print("-" * 85)
    
    for file_path in file_list:
        if is_excluded(file_path):
            continue
        try:
            df = pd.read_csv(file_path)
            if 'biopac' in df.columns and 'senstream' in df.columns:
                # Enak izsek kot pri navadni korelaciji, sicer metriki nista primerljivi.
                # reset_index, ker .shift() dela po poziciji in ne po casu.
                df = filter_out_initial_60_seconds(df).reset_index(drop=True)
                if df.empty:
                    if PRINT_FILE_ROWS:
                        print(f"{file_path.name:<45} | {'Empty After Trim':>10} | {'N/A':>8} | {'N/A':<10}")
                    continue

                best_corr = -1.0
                best_lag = 0
                best_pval = np.nan
                
                # Check shifts from -max_shift to +max_shift
                for s in range(-max_shift, max_shift + 1):
                    # Shift senstream relative to biopac
                    biopac_series = df['biopac']
                    senstream_shifted = df['senstream'].shift(s)
                    mask = ~(biopac_series.isna() | senstream_shifted.isna())
                    x_clean = biopac_series[mask].to_numpy()
                    y_clean = senstream_shifted[mask].to_numpy()
                    if len(x_clean) > 0:
                        corr, pval = spearmanr(x_clean, y_clean)
                        if not np.isnan(corr) and corr > best_corr:
                            best_corr = corr
                            best_lag = s
                            best_pval = pval  # Track the p-value for best correlation
                    
                
                if best_corr != -1.0:
                    results.append(best_corr)
                    lags.append(best_lag)
                    lag_sec = best_lag * 0.1
                    file_max_corrs[file_path.name] = (best_corr, lag_sec, best_pval)
                    
                    # Categorize
                    if best_corr >= 0.9:
                        cat = 'Great'
                        categories['Great (>= 0.9)'].append(file_path.name)
                    elif best_corr >= 0.7:
                        cat = 'Strong'
                        categories['Strong (>= 0.7)'].append(file_path.name)
                    elif best_corr >= 0.3:
                        cat = 'Medium'
                        categories['Medium (0.3 - 0.7)'].append(file_path.name)
                    else:
                        cat = 'Weak'
                        categories['Weak (< 0.3)'].append(file_path.name)
                        #results.pop() # Remove weak correlation from results for summary stats
                    
                    if PRINT_FILE_ROWS:
                        print(f"{file_path.name:<45} | {best_corr:>10.4f} | {lag_sec:>8.1f}s | {cat:<10}")
                else:
                    if PRINT_FILE_ROWS:
                        print(f"{file_path.name:<45} | {'N/A':>10} | {'N/A':>8} | {'N/A':<10}")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    if results:
        avg_corr = np.mean(results)
        max_corr = np.max(results)
        min_corr = np.min(results)
        
        print("\n" + "="*35)
        print("📈 MAX CROSS-CORRELATION SUMMARY")
        print("="*35)
        print(f"Total Files Analyzed: {len(results)}")
        print(f"Average Max Corr:     {avg_corr:.4f}")
        print(f"Best Max Corr:        {max_corr:.4f}")
        print(f"Worst Max Corr:       {min_corr:.4f}")
        print(f"Avg Absolute Lag:     {np.mean(np.abs(lags)) * 0.1:.2f}s")
        print("-" * 35)
        
        print("COUNTS BY CATEGORY (AFTER SHIFT):")
        for cat_name, files in categories.items():
            print(f"- {cat_name:<18}: {len(files)}")
            
        if categories['Weak (< 0.3)']:
            print("\n❌ STILL WEAK AFTER SHIFT (LIST):")
            for name in categories['Weak (< 0.3)']:
                print(f"  - {name}")
        
        print("="*35)
    return file_max_corrs


def output_summary_csv(file_corrs, max_corrs, csv_files, output_path):
    """Outputs a summary CSV of the correlation results."""
    rows = []
    for file_path in csv_files:
        if is_excluded(file_path):
            continue
        name = file_path.name
        
        metrics = file_corrs.get(name, {})
        std_corr = metrics.get("standard_spearman", np.nan)
        std_pval = metrics.get("p_value", np.nan)
        nrmse_range = metrics.get("nrmse_range", np.nan)
        nrmse_std = metrics.get("nrmse_std", np.nan)

        max_corr, lag, max_pval = max_corrs.get(name, (np.nan, np.nan, np.nan))
        rows.append({
            'file_name': name,
            'standard_spearman': std_corr,
            'standard_spearman_pval': std_pval,
            'max_spearman': max_corr,
            'max_spearman_pval': max_pval,
            'lag_seconds': lag,
            'raw_nrmse_range': nrmse_range,
            'raw_nrmse_std': nrmse_std,
        })

    df_out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\n Saved detailed raw correlation results to {output_path}")


def main():
    # Discovery logic
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent / "Meritve" / "Combined"
    output_csv = script_path.parent.parent / "Results" / "raw_correlation_all.csv"
    output_csv_sitting = script_path.parent.parent / "Results" / "raw_correlation_sitting.csv"
    output_csv_standing = script_path.parent.parent / "Results" / "raw_correlation_standing.csv"
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
    else:
        csv_files = filter_csv_paths(sorted(list(base_dir.rglob("*_combined.csv"))))
        csv_sitting = filter_csv_paths(sorted(list(base_dir.rglob("*1_combined.csv"))))
        csv_standing = filter_csv_paths(sorted(list(base_dir.rglob("*2_combined.csv"))))
        
        if not csv_files or not csv_sitting or not csv_standing:
            print(f"No combined CSV files found in {base_dir}")
        else:
            print("RUNNING STANDARD SPEARMAN CORRELATION...")
            std_results = calculate_correlations(csv_files)
            
            print("\nRUNNING MAX CROSS-CORRELATION (Shift Analysis)...")
            max_results = calculate_max_cross_correlations(csv_files)

            print(f"\n\n RUNNING STANDARD SPEARMAN CORRELATION ON SITTING")
            std_results_sitting = calculate_correlations(csv_sitting)

            print(f"\n\n RUNNING MAX CROSS-CORRELATION ON SITTING")
            max_results_sitting = calculate_max_cross_correlations(csv_sitting)

            print(f"\n\n RUNNING STANDARD SPEARMAN CORRELATION ON STANDING")
            std_results_standing = calculate_correlations(csv_standing)

            print(f"\n\n RUNNING MAX CROSS-CORRELATION ON STANDING")
            max_results_standing = calculate_max_cross_correlations(csv_standing)

            output_summary_csv(std_results, max_results, csv_files, output_csv)
            output_summary_csv(std_results_sitting, max_results_sitting, csv_sitting, output_csv_sitting)
            output_summary_csv(std_results_standing, max_results_standing, csv_standing, output_csv_standing)



            
if __name__ == "__main__":
    main()