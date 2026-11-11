import os
import glob
import pandas as pd
import scipy
import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np


def moving_average(signal, fs=10, window_seconds=1.0):
    window_samples = int(window_seconds * fs)

    if window_samples < 1:
        window_samples = 1

    return (
        pd.Series(signal)
        .rolling(window=window_samples, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
#For this script to work the _combined files must be made 
def get_all_csvs(path):
    # Get all CSV files under the folder and return relative paths
    csv_files = []
    file_names = []
    for root, _, files in os.walk(path):
        for file in files: 
            if file == "P018 2 movement test_combined.csv":
                continue

            if file.lower().endswith('combined.csv'):
                rel_path = os.path.relpath(os.path.join(path, root, file), start=path)
                csv_files.append(rel_path)
                file_names.append(file)
    return csv_files


def process_eda(eda_col):
    std = nk.standardize(eda_col)
    mva = moving_average(std,window_seconds=2)
    signals, info = nk.eda_process(mva, sampling_rate=10)
    #signals, info = nk.eda_process(eda_col, sampling_rate=10)
    peak_count = signals['SCR_Peaks'].sum()
    tonic_mean = signals['EDA_Tonic'].mean()
    average_amplitude = signals[signals['SCR_Peaks'] == 1]['SCR_Amplitude'].mean()
    summary = {
        'peaks': peak_count,
        'tonic_mean': tonic_mean,
        'avg_amp': average_amplitude
    }
    data = signals[['EDA_Phasic', 'EDA_Tonic', 'SCR_Peaks']].copy()
    data["EDA_Raw"] = eda_col
    return data, info, summary

def process_senstream_prefiltered(senstream_f_col):
    signals, info = nk.eda_process(nk.standardize(senstream_f_col), sampling_rate=10)
    #signals, info = nk.eda_process(senstream_f_col, sampling_rate=10)
    peak_count = signals['SCR_Peaks'].sum()
    return signals, info, peak_count

def compute_correlations(combined_df):
    raw_correlation = combined_df["biopac"].corr(combined_df["senstream"], method='spearman')
    tonic_correlation = combined_df["Biopac_Tonic"].corr(combined_df["Senstream_Tonic"], method='spearman')
    phasic_correlation = combined_df["Biopac_Phasic"].corr(combined_df["Senstream_Phasic"], method='spearman')
    correlation_summary = {
        'raw': raw_correlation,
        'tonic': tonic_correlation,
        'phasic': phasic_correlation
    }
    return correlation_summary

def plot_signals(combined_df, filename):
    phasic_signal = combined_df[["Biopac_Phasic", "Senstream_Phasic"]].copy()
    tonic_signal = combined_df[["Biopac_Tonic", "Senstream_Tonic"]].copy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(phasic_signal.index, phasic_signal['Senstream_Phasic'], label='Senstream Phasic')
    axes[0].plot(phasic_signal.index, phasic_signal['Biopac_Phasic'], label='Biopac Phasic')
    axes[0].set_title(f'Phasic EDA - {filename}')
    axes[0].set_ylabel('Phasic EDA')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    axes[1].plot(tonic_signal.index, tonic_signal['Senstream_Tonic'], label='Senstream Tonic')
    axes[1].plot(tonic_signal.index, tonic_signal['Biopac_Tonic'], label='Biopac Tonic')
    axes[1].set_title(f'Tonic EDA - {filename}')
    axes[1].set_ylabel('Tonic EDA')
    axes[1].set_xlabel('Sample')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def make_dataframe(data_bio, data_sen, original):
    result_df = pd.concat(
      [data_bio['EDA_Phasic'], data_bio['EDA_Tonic'], data_sen['EDA_Phasic'], data_sen['EDA_Tonic']], axis=1, keys=["Biopac_Phasic", "Biopac_Tonic", "Senstream_Phasic", "Senstream_Tonic"]  
    )
    combined_df = pd.concat([original, result_df], axis=1)
    return combined_df

def main():
    parent_dir = "Meritve/Filtered/neurokit2"
    os.makedirs(parent_dir, exist_ok=True)
    csvs = get_all_csvs('Meritve/Combined')
    results = []
    for i, file in enumerate(csvs):
        filename = os.path.basename(file)
        day_folder = os.path.basename(os.path.dirname(file))
        print(f"[{i+1}/{len(csvs)}] Processing: {filename}")
        df = pd.read_csv(file)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # Keep only first 120 seconds of data
        #df = df[df.index > 109 * 10]  # 120 seconds at 10Hz = 1200 samples
        #print(df.tail())
        
        senstream = df["senstream"].copy()
        senstream_f = df["senstream_filtered"].copy()
        biopac = df["biopac"].copy()
        time = df["time"].copy()
        sig_prefil, info_prefil, prefil_peaks = process_senstream_prefiltered(senstream_f)
        data_sen, info_sen, summary_sen = process_eda(senstream)
        data_bio, info_bio, summary_bio = process_eda(biopac)
        combined_df = make_dataframe(data_bio, data_sen, df)
        #plot_signals(combined_df, filename)
        correlation_summary = compute_correlations(combined_df)
        results.append({
            'File': filename,
            'Biopac_peaks' : summary_bio["peaks"],
            'Senstream_peaks': summary_sen["peaks"],
            'Senstream_prefiltered_peaks': prefil_peaks,
            'Biopac_tonic_mean': summary_bio["tonic_mean"],
            'Senstream_tonic_mean': summary_sen["tonic_mean"],
            'Biopac_average_amplitude': summary_bio["avg_amp"],
            'Senstream_average_amplitude': summary_sen["avg_amp"],
            'Correlation' : correlation_summary['raw'],
            'Tonic_Correlation' : correlation_summary['tonic'],
            'Phasic_Correlation' : correlation_summary['phasic']
        })
        output_dir =  f"{parent_dir}/{day_folder}"
        os.makedirs(output_dir, exist_ok=True)
        output_name = f"{output_dir}/{filename}"
        combined_df.to_csv(output_name, index=False)




    result_df = pd.DataFrame(results)
    output_name = "Meritve/Filtered/neurokit2/11eda_feature_summary_10hz.csv"
    #result_df.to_csv(output_name, index=False)
    print(f"""
          average correlation: {result_df['Correlation'].mean():.4f}
          average tonic correlation: {result_df['Tonic_Correlation'].mean():.4f}
          average phasic correlation: {result_df['Phasic_Correlation'].mean():.4f}
          """)
    
    #print(len(csvs))

if __name__ == "__main__":
    main()

