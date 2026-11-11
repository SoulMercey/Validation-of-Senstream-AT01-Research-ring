import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from file_exclusions import filter_csv_paths


roots = [
    os.path.join("Meritve", "Filtered", "cvxeda"),
    os.path.join("Meritve", "Filtered", "neurokit2"),
    os.path.join("Meritve", "Filtered", "manual"),
]
outputs = [
    os.path.join("Results", "cvxeda_quality_metrics_segmented.csv"),
    os.path.join("Results", "nk2_quality_metrics_segmented.csv"),
    os.path.join("Results", "manual_quality_metrics_segmented.csv"),
]

# The Senstream application exports only one processed signal.  We compare it
# with the BioPac phasic component obtained with NeuroKit2 and keep this as a
# separate result, because it is not a full tonic/phasic decomposition method.
APPLICATION_ROOT = os.path.join("Meritve", "Filtered", "neurokit2")
APPLICATION_OUTPUT = os.path.join(
    "Results", "application_phasic_quality_metrics_segmented.csv"
)


SIGNAL_PAIRS = {
    "Tonic": ("Biopac_Tonic", "Senstream_Tonic"),
    "Phasic": ("Biopac_Phasic", "Senstream_Phasic"),
    "SMNA": ("Biopac_SMNA", "Senstream_SMNA"),
}

BREAK_TIME_1_START = pd.to_timedelta(0, unit="s")
BREAK_TIME_1_END = pd.to_timedelta(60, unit="s")
BREAK_TIME_2_START = pd.to_timedelta(110, unit="s")
BREAK_TIME_2_END = pd.to_timedelta(160, unit="s")

SEGMENT_FILENAME_SUFFIX = {
    "static": "static",
    "dynamic": "dynamic",
    "break": "still",
}


def get_all_csvs(root_dir):
    csvs = []
    csvs_sitting = []
    csvs_standing = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                if "_combined" in file.lower():
                    csvs.append(os.path.join(root, file))
                if "1_combined" in file.lower():
                    csvs_sitting.append(os.path.join(root, file))
                elif "2_combined" in file.lower():
                    csvs_standing.append(os.path.join(root, file))

    csvs.sort()
    csvs_sitting.sort()
    csvs_standing.sort()
    csvs = filter_csv_paths(csvs)
    csvs_sitting = filter_csv_paths(csvs_sitting)
    csvs_standing = filter_csv_paths(csvs_standing)
    return csvs, csvs_sitting, csvs_standing


def get_segment_masks(df: pd.DataFrame) -> dict:
    elapsed = pd.to_datetime(df["time"], errors="coerce") - pd.to_datetime(df["time"], errors="coerce").iloc[0]
    return {
        "static": (elapsed > BREAK_TIME_1_END) & (elapsed <= BREAK_TIME_2_START),
        "dynamic": (elapsed > BREAK_TIME_2_END),
        "break": ((elapsed > BREAK_TIME_2_START) & (elapsed < BREAK_TIME_2_END)),
    }


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

    # normalized RMSE by BioPack range
    x_range = np.max(x) - np.min(x)
    nrmse_range = rmse / x_range if x_range != 0 else np.nan

    # normalized RMSE by BioPack std
    x_std = np.std(x, ddof=1)
    nrmse_std = rmse / x_std if x_std != 0 else np.nan

    return mae, rmse, nrmse_range, nrmse_std

def compute_max_cross_corr(df: pd.DataFrame, col_bio: str, col_sen: str, max_shift: int = 100) -> tuple[float, int, float]:
    """Compute max cross-correlation with lag and p-value."""
    best_corr = -1.0
    best_lag = 0
    best_pval = np.nan

    for shift in range(-max_shift, max_shift + 1):
        bio_series = df[col_bio]
        sen_shifted = df[col_sen].shift(shift)
        mask = ~(bio_series.isna() | sen_shifted.isna())
        x_clean = bio_series[mask].to_numpy()
        y_clean = sen_shifted[mask].to_numpy()

        if len(x_clean) > 0:
            corr, pval = spearmanr(x_clean, y_clean)
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_lag = shift
                best_pval = pval

    if best_corr == -1.0:
        return np.nan, 0, np.nan

    return best_corr, best_lag, best_pval


def compute_pair_metrics(df, col_bio, col_sen):
    clean = df[[col_bio, col_sen]].dropna()

    x = clean[col_bio].to_numpy()
    y = clean[col_sen].to_numpy()

    if len(x) < 3:
        spearman, spearman_pval = np.nan, np.nan
    else:
        spearman, spearman_pval = spearmanr(x, y)

    max_spearman, max_lag, max_spearman_pval = compute_max_cross_corr(clean, col_bio, col_sen)

    mae, rmse, nrmse_range, nrmse_std = error_metrics(x, y)
    ba_bias, ba_lower, ba_upper = bland_altman_metrics(x, y)

    return {
        "Spearman": spearman,
        "Spearman_pval": spearman_pval,
        "Max_Spearman": max_spearman,
        "Max_Spearman_pval": max_spearman_pval,
        "Lag_Seconds": max_lag * 0.1,
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE_Range": nrmse_range,
        "NRMSE_STD": nrmse_std,
        "BA_Bias": ba_bias,
        "BA_Lower_LOA": ba_lower,
        "BA_Upper_LOA": ba_upper,
    }

def summarise_metrics():
    for file in outputs:
        print(f"\nAverage correlations for file {file}:")
        result_df = pd.read_csv(file)
        for signal_name in SIGNAL_PAIRS.keys():
            spearman_col = f"{signal_name}_Spearman"
            max_spearman_col = f"{signal_name}_Max_Spearman"
            lag_col = f"{signal_name}_Lag_Seconds"

            if spearman_col in result_df.columns:
                print(f"{signal_name} Spearman: {result_df[spearman_col].mean():.4f}")
            if max_spearman_col in result_df.columns:
                print(f"{signal_name} Max Spearman: {result_df[max_spearman_col].mean():.4f}")
            if lag_col in result_df.columns:
                print(f"{signal_name} Avg Lag (s): {result_df[lag_col].mean():.4f}")

        print("\nAverage Bland-Altman bias:")
        for signal_name in SIGNAL_PAIRS.keys():
            bias_col = f"{signal_name}_BA_Bias"
            if bias_col in result_df.columns:
                print(f"{signal_name} bias: {result_df[bias_col].mean():.4f}")


def process_and_export(csv_files, output_file):
    results = []

    for i, file in enumerate(csv_files):
        filename = os.path.basename(file)
        day_folder = os.path.basename(os.path.dirname(file))

        print(f"[{i + 1}/{len(csv_files)}] Processing: {day_folder}/{filename}")

        df = pd.read_csv(file)
        segment_masks = get_segment_masks(df)

        for segment_name, mask in segment_masks.items():
            segment_df = df[mask].copy()
            if segment_df.empty:
                continue

            segment_filename = f"{os.path.splitext(filename)[0]}_{SEGMENT_FILENAME_SUFFIX[segment_name]}.csv"

            row = {
                "Day": day_folder,
                "File": segment_filename,
                "Segment": segment_name,
            }

            for signal_name, (bio_col, sen_col) in SIGNAL_PAIRS.items():
                if bio_col not in segment_df.columns or sen_col not in segment_df.columns:
                    print(f"  Missing columns for {signal_name} in {segment_name}, skipping")
                    continue

                metrics = compute_pair_metrics(segment_df, bio_col, sen_col)

                for metric_name, value in metrics.items():
                    row[f"{signal_name}_{metric_name}"] = value

            results.append(row)

    result_df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)

    print("\nSaved:")
    print(output_file)

    print("\nAverage correlations:")
    for signal_name in SIGNAL_PAIRS.keys():
        spearman_col = f"{signal_name}_Spearman"
        max_spearman_col = f"{signal_name}_Max_Spearman"
        lag_col = f"{signal_name}_Lag_Seconds"

        if spearman_col in result_df.columns:
            print(f"{signal_name} Spearman: {result_df[spearman_col].mean():.4f}")
        if max_spearman_col in result_df.columns:
            print(f"{signal_name} Max Spearman: {result_df[max_spearman_col].mean():.4f}")
        if lag_col in result_df.columns:
            print(f"{signal_name} Avg Lag (s): {result_df[lag_col].mean():.4f}")

    print("\nAverage Bland-Altman bias:")
    for signal_name in SIGNAL_PAIRS.keys():
        bias_col = f"{signal_name}_BA_Bias"
        if bias_col in result_df.columns:
            print(f"{signal_name} bias: {result_df[bias_col].mean():.4f}")

    return result_df


def process_application_and_export(csv_files, output_file):
    """Calculate segmented correlations for the Senstream app output.

    The app's ``senstream_filtered`` column is treated as its phasic signal.
    The reference is the BioPac phasic component from the NeuroKit2 export.
    """
    results = []

    for i, file in enumerate(csv_files):
        filename = os.path.basename(file)
        day_folder = os.path.basename(os.path.dirname(file))
        print(f"[{i + 1}/{len(csv_files)}] Processing app output: {day_folder}/{filename}")

        df = pd.read_csv(file)
        segment_masks = get_segment_masks(df)

        for segment_name, mask in segment_masks.items():
            segment_df = df[mask].copy()
            if segment_df.empty:
                continue

            required = ["Biopac_Phasic", "senstream_filtered"]
            if any(column not in segment_df.columns for column in required):
                raise ValueError(
                    f"Missing application columns in {file}: {required}"
                )

            metrics = compute_pair_metrics(
                segment_df, "Biopac_Phasic", "senstream_filtered"
            )
            row = {
                "Day": day_folder,
                "File": f"{os.path.splitext(filename)[0]}_{SEGMENT_FILENAME_SUFFIX[segment_name]}.csv",
                "Segment": segment_name,
            }
            row.update({f"Phasic_{name}": value for name, value in metrics.items()})
            results.append(row)

    result_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"\nSaved application results: {output_file}")
    return result_df


def main():
    for root, output in zip(roots, outputs):
        ROOT_DIR = root
        OUTPUT_FILE = output
        csvs, csvs_sitting, csvs_standing = get_all_csvs(ROOT_DIR)

        process_and_export(csvs, OUTPUT_FILE)

        if csvs_sitting:
            sitting_output = OUTPUT_FILE.replace("quality_metrics_segmented.csv", "quality_metrics_segmented_sitting.csv")
            process_and_export(csvs_sitting, sitting_output)

        if csvs_standing:
            standing_output = OUTPUT_FILE.replace("quality_metrics_segmented.csv", "quality_metrics_segmented_standing.csv")
            process_and_export(csvs_standing, standing_output)

    app_csvs, app_csvs_sitting, app_csvs_standing = get_all_csvs(APPLICATION_ROOT)
    process_application_and_export(app_csvs, APPLICATION_OUTPUT)
    if app_csvs_sitting:
        process_application_and_export(
            app_csvs_sitting,
            APPLICATION_OUTPUT.replace(
                "quality_metrics_segmented.csv",
                "quality_metrics_segmented_sitting.csv",
            ),
        )
    if app_csvs_standing:
        process_application_and_export(
            app_csvs_standing,
            APPLICATION_OUTPUT.replace(
                "quality_metrics_segmented.csv",
                "quality_metrics_segmented_standing.csv",
            ),
        )

    summarise_metrics()


if __name__ == "__main__":
    main()
