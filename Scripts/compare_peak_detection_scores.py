import pandas as pd
import numpy as np
import neurokit2 as nk
import os
#from neurokit_filtering_export import get_all_csvs
import matplotlib.pyplot as plt
from pathlib import Path

from file_exclusions import filter_csv_paths
from signal_utils import trim_initial_seconds


def get_all_csvs(type=None):
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent / "Meritve" / "Filtered"
    if type == "manual":
        base_dir = base_dir / "manual"
    elif type == "neurokit2":
        base_dir = base_dir / "neurokit2"
    elif type == "cvxeda":
        base_dir = base_dir / "cvxeda"

    csv = sorted(list(base_dir.rglob("*_combined.csv")))
    csv_sitting = sorted(list(base_dir.rglob("*1_combined.csv")))
    csv_standing = sorted(list(base_dir.rglob("*2_combined.csv")))
    csv = filter_csv_paths(csv)
    csv_sitting = filter_csv_paths(csv_sitting)
    csv_standing = filter_csv_paths(csv_standing)
    return csv, csv_sitting, csv_standing


def check_name_pairity(manual_csvs, neurokit_csvs, cvxeda_csvs):
    for i,csv in enumerate(manual_csvs):
        name_manual = os.path.basename(csv)
        name_neurokit = os.path.basename(neurokit_csvs[i])
        name_cvxeda = os.path.basename(cvxeda_csvs[i])
        if name_manual != name_neurokit or name_manual != name_cvxeda:
            print(f"Filename mismatch at index {i}:")
            print(f"  Manual: {name_manual}")
            print(f"  Neurokit: {name_neurokit}")
            print(f"  CVXEDA: {name_cvxeda}")
            raise ValueError(f"""
                             Filename mismatch at index {i}
                             Manual: {name_manual}
                             Neurokit: {name_neurokit}
                             CVXEDA: {name_cvxeda}
                            """)
        else:
            pass
            #print(f"Filenames match at index {i}: {name_manual}")

def get_phasic_peaks(df_manual, df_neurokit, df_cvxeda):
    """
    Extract peaks from filtered data with nk2 function returns 3 dictionaries with keys 'Biopac' and 'Senstream' for each method
    each of those is also a dictionary with "SCR_Amplitude", "SCR_Onsets", and "SCR_Peaks" respectively.

    """
    manual_peaks_bio = nk.eda_findpeaks(df_manual['Biopac_Phasic'])
    manual_peaks_sen = nk.eda_findpeaks(df_manual['Senstream_Phasic'])
    neurokit_peaks_bio = nk.eda_findpeaks(df_neurokit['Biopac_Phasic'])
    neurokit_peaks_sen = nk.eda_findpeaks(df_neurokit['Senstream_Phasic'])
    cvxeda_peaks_bio = nk.eda_findpeaks(df_cvxeda['Biopac_Phasic'])
    cvxeda_peaks_sen = nk.eda_findpeaks(df_cvxeda['Senstream_Phasic'])
    prefiltered_peaks_sen = nk.eda_findpeaks(df_cvxeda['senstream_filtered'])
    manual_peaks = {
        'Biopac': manual_peaks_bio,
        'Senstream': manual_peaks_sen,
    }
    neurokit_peaks = {
        'Biopac': neurokit_peaks_bio,
        'Senstream': neurokit_peaks_sen,
    }
    cvxeda_peaks = {
        'Biopac': cvxeda_peaks_bio,
        'Senstream': cvxeda_peaks_sen,
    }
    prefiltered_peaks_man = {
        'Senstream': prefiltered_peaks_sen,
        'Biopac': manual_peaks_bio,
    }
    prefiltered_peaks_neuro = {
        'Senstream': prefiltered_peaks_sen,
        'Biopac': neurokit_peaks_bio,
    }
    prefiltered_peaks_cvx = {
        'Senstream': prefiltered_peaks_sen,
        'Biopac': cvxeda_peaks_bio,
    }
    #print(f"""
    #Manual peak count is Biopac: {len(manual_peaks_bio['SCR_Peaks'])} | Senstream: {len(manual_peaks_sen['SCR_Peaks'])}
    #Neurokit peak count is Biopac: {len(neurokit_peaks_bio['SCR_Peaks'])} | Senstream: {len(neurokit_peaks_sen['SCR_Peaks'])}
    #CVXEDA peak count is Biopac: {len(cvxeda_peaks_bio['SCR_Peaks'])} | Senstream: {len(cvxeda_peaks_sen['SCR_Peaks'])}
    #""")

    return manual_peaks, neurokit_peaks, cvxeda_peaks, prefiltered_peaks_man, prefiltered_peaks_neuro, prefiltered_peaks_cvx


def _plot_phasic_method(ax, df, peaks, title):
    biopac_phasic = df['Biopac_Phasic']
    senstream_phasic = df['Senstream_Phasic']
    biopac_peak_idx = peaks['Biopac']['SCR_Peaks']
    senstream_peak_idx = peaks['Senstream']['SCR_Peaks']
    time = pd.to_datetime(df['time']) 
    ax.plot(time, biopac_phasic, label='Biopac Phasic', color='tab:blue')
    ax.plot(time, senstream_phasic, label='Senstream Phasic', color='tab:orange')

    ax.scatter(
        time.iloc[biopac_peak_idx],
        biopac_phasic.iloc[biopac_peak_idx],
        label='Biopac Peaks',
        color='tab:blue',
        marker='o',
        s=25,
        edgecolors='black',
        linewidths=0.5,
        zorder=3
    )
    ax.scatter(
        time.iloc[senstream_peak_idx],
        senstream_phasic.iloc[senstream_peak_idx],
        label='Senstream Peaks',
        color='tab:orange',
        marker='x',
        s=35,
        linewidths=1.5,
        zorder=3
    )

    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Phasic EDA')
    ax.grid(True)
    ax.legend(loc='upper right')


def plot_all_phasic_components(
    df_manual,
    df_neurokit,
    df_cvxeda,
    manual_peaks=None,
    neurokit_peaks=None,
    cvxeda_peaks=None,
    filename=None
):
    """
    Show all phasic components in a 2x2 figure and mark detected SCR peaks.

    Each method panel contains both Biopac_Phasic and Senstream_Phasic signals.
    Peaks are taken from get_phasic_peaks() or computed if they are not provided.
    """
    if manual_peaks is None or neurokit_peaks is None or cvxeda_peaks is None:
        manual_peaks, neurokit_peaks, cvxeda_peaks = get_phasic_peaks(
            df_manual,
            df_neurokit,
            df_cvxeda
        )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    axes = axes.flatten()

    _plot_phasic_method(axes[0], df_manual, manual_peaks, 'Manual Filtering')
    _plot_phasic_method(axes[1], df_neurokit, neurokit_peaks, 'NeuroKit2 Filtering')
    _plot_phasic_method(axes[2], df_cvxeda, cvxeda_peaks, 'CVXEDA Filtering')
    axes[3].axis('off')

    if filename:
        fig.suptitle(f'Phasic Components and Peaks - {filename}', fontsize=14)

    plt.tight_layout()
    plt.show()

def match_peaks(peaks, signal_length=None, tolerance_window_samples=10):
    """
    Return binary arrays for matched Biopac peaks and unmatched Senstream peaks.

    matched_scr_peaks has 1 at Biopac peak locations that have a Senstream
    peak within +/- tolerance_window_samples. unmatched_scr_peaks has 1 at
    Senstream peak locations that were not matched to a Biopac peak.
    """
    biopac_peaks = np.asarray(peaks['Biopac']['SCR_Peaks'], dtype=int)
    senstream_peaks = np.asarray(peaks['Senstream']['SCR_Peaks'], dtype=int)

    if signal_length is None:
        max_peak = -1
        if len(biopac_peaks) > 0:
            max_peak = max(max_peak, biopac_peaks.max())
        if len(senstream_peaks) > 0:
            max_peak = max(max_peak, senstream_peaks.max())
        signal_length = max_peak + 1

    matched_scr_peaks = np.zeros(signal_length, dtype=int)
    peak_count = {
        'Biopac': len(biopac_peaks),
        'Senstream': len(senstream_peaks)
    }

    used_senstream_peaks = set()
    timing_errors = []
    for b_peak in biopac_peaks:
        candidate_matches = [
            s_peak for s_peak in senstream_peaks
            if s_peak not in used_senstream_peaks
            and abs(b_peak - s_peak) <= tolerance_window_samples
        ]

        if candidate_matches:
            best_match = min(candidate_matches, key=lambda s_peak: abs(b_peak - s_peak))
            used_senstream_peaks.add(best_match)
            matched_scr_peaks[b_peak] = 1
            # 10 Hz sampling rate, so divide by 10 to get error in seconds
            timing_errors.append(abs(b_peak - best_match) / 10.0)

    unmatched_scr_peaks = np.zeros(signal_length, dtype=int)
    for s_peak in senstream_peaks:
        if s_peak not in used_senstream_peaks:
            unmatched_scr_peaks[s_peak] = 1

    peak_count_error = abs(peak_count["Biopac"] - peak_count["Senstream"])
    recall = matched_scr_peaks.sum() / peak_count['Biopac'] if peak_count['Biopac'] > 0 else 0
    precision = matched_scr_peaks.sum() / peak_count['Senstream'] if peak_count['Senstream'] > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # NaN in ne 0.0: meritev brez ujemanj nima casovne napake in ne sme prispevati
    # popolne vrednosti 0 s v povprecje
    avg_timing_error = np.mean(timing_errors) if timing_errors else np.nan

    scores = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'peak_count_error': peak_count_error,
        'avg_timing_error': avg_timing_error
    }
    return matched_scr_peaks, unmatched_scr_peaks, scores
    

def plot_matched_peaks(df, matched_peaks, unmatched_peaks, filename=None):
    time = pd.to_datetime(df['time'])
    plt.figure(figsize=(12, 4))
    for idx in np.where(matched_peaks == 1)[0]:
        plt.axvline(time.iloc[idx], color='red', alpha=0.5, linestyle='--', linewidth=1)
    for idx in np.where(unmatched_peaks == 1)[0]:
        plt.axvline(time.iloc[idx], color='black', alpha=0.5, linewidth=1)
    plt.plot(time, df['Biopac_Phasic'], label='Biopac Phasic', color='tab:blue')
    plt.plot(time, df['Senstream_Phasic'], label='Senstream Phasic', color='tab:orange')
    title = 'Matched Peaks'
    if filename:
        title = f'{filename} - {title}'
    plt.title(title)
    plt.legend(loc='upper right')
    plt.xlabel('Time')
    plt.ylabel('Phasic EDA')
    plt.grid(True)
    plt.show()


def compare_peak_detection_scores(manual_csvs, neurokit_csvs, cvxeda_csvs, case_name="all", tolerance_window_samples=10, plot=False):
    """
    Run peak-matching comparison for one set of corresponding CSV files.

    This helper can be invoked from main for the full, sitting, or standing cases.
    """
    if not (len(manual_csvs) == len(neurokit_csvs) == len(cvxeda_csvs)):
        raise ValueError(
            f"CSV list lengths do not match for {case_name}: "
            f"manual={len(manual_csvs)}, neurokit={len(neurokit_csvs)}, cvxeda={len(cvxeda_csvs)}"
        )

    check_name_pairity(manual_csvs, neurokit_csvs, cvxeda_csvs)

    scores_manual = []
    scores_neurokit = []
    scores_cvxeda = []
    scores_prefiltered_man = []
    scores_prefiltered_cvx = []
    scores_prefiltered_nk = []

    detailed_rows = []

    for i, csv in enumerate(manual_csvs):
        name_manual = os.path.basename(csv)
        name_neurokit = os.path.basename(neurokit_csvs[i])
        name_cvxeda = os.path.basename(cvxeda_csvs[i])
        # print(f"[{case_name}] [{i + 1}/{len(manual_csvs)}] Processing: {name_manual}")

        # Enak izsek kot pri analizi surovih signalov: prvih 60 s je aklimatizacija
        df_manual = trim_initial_seconds(pd.read_csv(csv))
        df_neurokit = trim_initial_seconds(pd.read_csv(neurokit_csvs[i]))
        df_cvxeda = trim_initial_seconds(pd.read_csv(cvxeda_csvs[i]))

        manual_peaks, neurokit_peaks, cvxeda_peaks, prefiltered_peaks_man, prefiltered_peaks_neuro, prefiltered_peaks_cvx = get_phasic_peaks(df_manual, df_neurokit, df_cvxeda)

        def process_method(peaks, df, method_name, scores_list):
            _, _, scores = match_peaks(peaks, signal_length=len(df), tolerance_window_samples=tolerance_window_samples)
            scores_list.append(scores)
            detailed_rows.append({
                "file": name_manual,
                "method": method_name,
                **scores
            })

        process_method(manual_peaks, df_manual, "manual", scores_manual)
        process_method(neurokit_peaks, df_neurokit, "neurokit", scores_neurokit)
        process_method(cvxeda_peaks, df_cvxeda, "cvxeda", scores_cvxeda)
        process_method(prefiltered_peaks_man, df_manual, "prefiltered_manual", scores_prefiltered_man)
        process_method(prefiltered_peaks_neuro, df_neurokit, "prefiltered_neurokit", scores_prefiltered_nk)
        process_method(prefiltered_peaks_cvx, df_cvxeda, "prefiltered_cvxeda", scores_prefiltered_cvx)

        if plot:
            plot_matched_peaks(df_manual, np.zeros(len(df_manual), dtype=int), np.zeros(len(df_manual), dtype=int), filename=name_manual)

    # Save detailed per-file results to CSV
    detailed_df = pd.DataFrame(detailed_rows)
    results_dir = Path(__file__).resolve().parent.parent / "Results"
    os.makedirs(results_dir, exist_ok=True)
    tol_sec = tolerance_window_samples / 10.0
    detailed_csv_path = results_dir / f"peak_detection_{case_name}.csv"
    detailed_df.to_csv(detailed_csv_path, index=False)

    summary_rows = []
    for method_name, scores_list in {
        "manual": scores_manual,
        "neurokit": scores_neurokit,
        "cvxeda": scores_cvxeda,
        "prefiltered_manual": scores_prefiltered_man,
        "prefiltered_neurokit": scores_prefiltered_nk,
        "prefiltered_cvxeda": scores_prefiltered_cvx,
    }.items():
        if not scores_list:
            continue
        metrics = pd.DataFrame(scores_list).mean().to_dict()
        summary_rows.append({
            "case": case_name,
            "method": method_name,
            **metrics,
        })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df[[
            "case", "method", "precision", "recall", "f1_score", 
            "peak_count_error", "avg_timing_error"
        ]]
        summary_df = summary_df.sort_values(["f1_score", "precision", "recall"], ascending=False)

    return summary_df


def summarize_scores(summary_df):
    if summary_df.empty:
        print("No summary data available.")
        return summary_df

    #print(summary_df.to_string(index=False))
    return summary_df


def main():
    manual_csvs, manual_sitting, manual_standing = get_all_csvs("manual")
    neurokit_csvs, neurokit_sitting, neurokit_standing = get_all_csvs("neurokit2")
    cvxeda_csvs, cvxeda_sitting, cvxeda_standing = get_all_csvs("cvxeda")

    print(f"Found {len(manual_csvs)} manual CSVs, {len(neurokit_csvs)} neurokit CSVs, and {len(cvxeda_csvs)} cvxeda CSVs.")

    # We want to test sensitivity to tolerance windows of +/- 0.5s (5), +/- 1.0s (10), +/- 2.0s (20)
    tol = 10
    all_summaries = []

    summary_all = compare_peak_detection_scores(manual_csvs, neurokit_csvs, cvxeda_csvs, case_name="all", tolerance_window_samples=tol)
    summary_sitting = compare_peak_detection_scores(manual_sitting, neurokit_sitting, cvxeda_sitting, case_name="sitting", tolerance_window_samples=tol)
    summary_standing = compare_peak_detection_scores(manual_standing, neurokit_standing, cvxeda_standing, case_name="standing", tolerance_window_samples=tol)
    


if __name__ == "__main__":
    main()
    







    
