import os
import glob
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np
from neurokit_filtering_export import get_all_csvs, moving_average
import contextlib
from scipy.signal import butter, filtfilt


def standardize_eda(combined_df):
    """Z-score the two raw EDA signals without importing the cvxEDA pipeline."""
    result = combined_df[["time", "biopac", "senstream"]].copy()
    for column in ("biopac", "senstream"):
        values = result[column]
        result[column] = (values - values.mean()) / values.std()
    return result


def compute_correlations(combined_df):
    return {
        "raw": combined_df["biopac"].corr(combined_df["senstream"], method="spearman"),
        "tonic": combined_df["Biopac_Tonic"].corr(combined_df["Senstream_Tonic"], method="spearman"),
        "phasic": combined_df["Biopac_Phasic"].corr(combined_df["Senstream_Phasic"], method="spearman"),
    }


def butter_lowpass(signal, fs=10, cutoff=0.05, order=2):
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist

    b, a = butter(
        order,
        normal_cutoff,
        btype="low",
        analog=False
    )

    return filtfilt(b, a, signal)

def plot_signals(df, filename=None):
    time = pd.to_datetime(df["time"], errors="coerce")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()

    axes[0].plot(time, df["biopac"], label="Biopac", color="black", linewidth=0.8)
    axes[0].plot(time, df["biopac_smooth"], label="Biopac Smooth", color="orange", linewidth=1.2)
    axes[0].set_title("Biopac")

    axes[1].plot(time, df["senstream"], label="Senstream", color="black", linewidth=0.8)
    axes[1].plot(time, df["senstream_smooth"], label="Senstream Smooth", color="blue", linewidth=1.2)
    axes[1].set_title("Senstream")
    axes[1].set_xlabel("Time")
    axes[2].plot(time, df["biopac_tonic"], label="Biopac Tonic", color="red")
    axes[2].plot(time, df["senstream_tonic"], label="Senstream Tonic", color="blue")
    axes[2].set_title("Tonic")
    axes[2].set_xlabel("Time")


    axes[3].plot(time, df["biopac_phasic"], label="Biopac Phasic", color="red")
    axes[3].plot(time, df["senstream_phasic"], label="Senstream Phasic", color="blue")
    axes[3].set_title("Phasic")
    axes[3].set_xlabel("Time")

    for ax in axes[:4]:
        ax.set_ylabel("EDA")
        ax.legend(loc="upper right")
        ax.grid(True)

    title = "Manual EDA filtering"
    if filename is not None:
        title = f"{filename} - {title}"
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def process_eda_manual(df, filename=None):
    df_std = standardize_eda(df)
    senstream = df_std["senstream"].copy()
    biopac = df_std["biopac"].copy()

    sen_smooth = moving_average(senstream.values, window_seconds=2)
    bio_smooth = moving_average(biopac.values, window_seconds=2)

    #get tonic component
    bio_tonic = butter_lowpass(bio_smooth)
    sen_tonic = butter_lowpass(sen_smooth)

    #phasic = smooth-tonic
    bio_phasic = bio_smooth - bio_tonic
    sen_phasic = sen_smooth - sen_tonic

    result = pd.DataFrame({
        "time": df["time"].values,
        "biopac": biopac.values,
        "senstream": senstream.values,
        "senstream_filtered": df["senstream_filtered"].values,
        "Biopac_Phasic": bio_phasic,
        "Biopac_Tonic": bio_tonic,
        "Biopac_Smooth": bio_smooth,
        "Senstream_Phasic": sen_phasic,
        "Senstream_Tonic": sen_tonic,
        "Senstream_Smooth": sen_smooth,
    })

    #plot_signals(result, filename=filename)
    return result

    

def main():
    parent_dir = "Meritve/Filtered/manual"
    os.makedirs(parent_dir, exist_ok=True)
    csvs = get_all_csvs('Meritve/Combined')
    results = []
    for i, file in enumerate(csvs):
        filename = os.path.basename(file)
        day_folder = os.path.basename(os.path.dirname(file))
        print(f"[{i+1}/{len(csvs)}] Processing: {filename}")
        df = pd.read_csv(file)
        df_processed = process_eda_manual(df, filename=filename)
        correlation_summary = compute_correlations(df_processed)
        results.append({
            'File': filename,
            'Correlation' : correlation_summary['raw'],
            'Tonic_Correlation' : correlation_summary['tonic'],
            'Phasic_Correlation' : correlation_summary['phasic']
        })
        output_dir =  f"{parent_dir}/{day_folder}"
        os.makedirs(output_dir, exist_ok=True)
        df_processed.to_csv(f"{output_dir}/{filename}", index=False)
    result_df = pd.DataFrame(results)
    print(f"""
          average correlation: {result_df['Correlation'].mean():.4f}
          average tonic correlation: {result_df['Tonic_Correlation'].mean():.4f}
          average phasic correlation: {result_df['Phasic_Correlation'].mean():.4f}
          """)
    output_name = "Meritve/Filtered/manual/11eda_feature_summary_10hz.csv"
    result_df.to_csv(output_name, index=False)

if __name__ == "__main__":
    main()
