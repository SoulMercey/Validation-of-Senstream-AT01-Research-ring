import pandas as pd
import os
import cvxeda as cvx
import matplotlib.pyplot as plt
from neurokit_filtering_export import get_all_csvs, moving_average
import contextlib

def standardize_eda(combined_df):
    biopac = combined_df["biopac"].copy().values
    senstream = combined_df["senstream"].copy().values
    biopac_standardized = (biopac - biopac.mean()) / biopac.std()
    senstream_standardized = (senstream - senstream.mean()) / senstream.std()
    standardized = pd.DataFrame({
        "time": combined_df["time"].values,
        "biopac": biopac_standardized,
        "senstream": senstream_standardized,
    })
    return standardized

def plot_orig_std(df, df_std):
    # Plot original and standardized signals: two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df["time"], df["biopac"], label="original biopac", alpha=0.7)
    axes[0].plot(df_std["time"], df_std["biopac"], label="standardized biopac", alpha=0.9)
    axes[0].set_ylabel("Biopac")
    axes[0].set_title("Biopac: original vs standardized")
    axes[0].legend()

    axes[1].plot(df["time"], df["senstream"], label="original senstream", alpha=0.7)
    axes[1].plot(df_std["time"], df_std["senstream"], label="standardized senstream", alpha=0.9)
    axes[1].set_ylabel("Senstream")
    axes[1].set_title("Senstream: original vs standardized")
    axes[1].legend()

    axes[1].set_xlabel("time")
    fig.tight_layout()
    plt.show()

def get_delta_time(df):
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    delta_time = df["time"][1]-df["time"][0]
    #print(f"Delta time  is {delta_time.total_seconds()} s")
    return delta_time.total_seconds()

def run_cvxeda_silent(signal, dt):
    #cvx.cvxEDA has unsupressable output of calculations so we made this helper for a cleaner output
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return cvx.cvxEDA(signal, dt)

def process_eda_cvx(df):
    df_std = standardize_eda(df)
    delta_time = get_delta_time(df_std)
    #process biopac
    df_std["biopac"] = moving_average(df_std["biopac"].values, window_seconds=2)
    phasic_b, smna_b, tonic_b, _, _, e_b, _ = run_cvxeda_silent(df_std["biopac"].values, delta_time)
    #process senstream
    df_std["senstream"] = moving_average(df_std["senstream"].values, window_seconds=2)
    phasic_s, smna_s, tonic_s, _, _, e_s, _ = run_cvxeda_silent(df_std["senstream"].values, delta_time)

    result = pd.DataFrame({
        "time": df["time"].values,
        "biopac": df["biopac"].values,
        "senstream": df["senstream"].values,
        "senstream_filtered": df["senstream_filtered"].values,
        "Biopac_Phasic": phasic_b,
        "Biopac_Tonic": tonic_b,
        "Biopac_SMNA": smna_b,
        "Senstream_Phasic": phasic_s,
        "Senstream_Tonic": tonic_s,
        "Senstream_SMNA": smna_s
    })
    return result, e_b, e_s

def compute_correlations(combined_df):
    corr_type = "spearman"
    raw_correlation = combined_df["biopac"].corr(combined_df["senstream"], method=corr_type)
    tonic_correlation = combined_df["Biopac_Tonic"].corr(combined_df["Senstream_Tonic"], method=corr_type)
    phasic_correlation = combined_df["Biopac_Phasic"].corr(combined_df["Senstream_Phasic"], method=corr_type)
    correlation_summary = {
        'raw': raw_correlation,
        'tonic': tonic_correlation,
        'phasic': phasic_correlation
    }
    return correlation_summary


def plot_phasic_and_tonic(df_processed):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    axes[0, 0].plot(df_processed["time"], df_processed["Biopac_Phasic"], label="Biopac Phasic", color="tab:blue")
    axes[0, 0].plot(df_processed["time"], df_processed["Senstream_Phasic"], label="Senstream Phasic", color="tab:red")
    axes[0, 0].set_title("Phasic components")
    axes[0, 0].set_ylabel("Phasic amplitude")
    axes[0, 0].legend()

    axes[0, 1].plot(df_processed["time"], df_processed["Biopac_Tonic"], label="Biopac Tonic", color="tab:blue")
    axes[0, 1].plot(df_processed["time"], df_processed["Senstream_Tonic"], label="Senstream Tonic", color="tab:red")
    axes[0, 1].set_title("Tonic components")
    axes[0, 1].set_ylabel("Tonic amplitude")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].legend()

    axes[1, 0].plot(df_processed["time"], df_processed["biopac"], label="Biopac Raw", color="tab:blue")
    axes[1, 0].plot(df_processed["time"], df_processed["senstream"], label="Senstream Raw", color="tab:red")
    axes[1, 0].set_title("Raw data")
    axes[1, 0].set_ylabel(r"Conductance ($\mu$S)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].legend()

    axes[1, 1].plot(df_processed["time"], df_processed["Biopac_SMNA"], label="Biopac SMNA", color="tab:blue")
    axes[1, 1].plot(df_processed["time"], df_processed["Senstream_SMNA"], label="Senstream SMNA", color="tab:red", alpha = 0.5)
    axes[1, 1].set_title("sparse SMNA driver")
    axes[1, 1].set_ylabel("SMNA level")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].legend()

    fig.tight_layout()
    plt.show()

def main():
    parent_dir = "Meritve/Filtered/cvxeda"
    os.makedirs(parent_dir, exist_ok=True)
    csvs = get_all_csvs('Meritve/Combined')
    results = []
    for i, file in enumerate(csvs):
        filename = os.path.basename(file)
        day_folder = os.path.basename(os.path.dirname(file))
        print(f"[{i+1}/{len(csvs)}] Processing: {filename}")
        df = pd.read_csv(file)
        #print(df.columns)
        df_processed, e_b, e_s = process_eda_cvx(df)
        df_processed["time"] = pd.to_datetime(df_processed["time"])
        correlation_summary = compute_correlations(df_processed)
        results.append({
            'File': filename,
            'Correlation' : correlation_summary['raw'],
            'Tonic_Correlation' : correlation_summary['tonic'],
            'Phasic_Correlation' : correlation_summary['phasic']
        })
        output_dir =  f"{parent_dir}/{day_folder}"
        os.makedirs(output_dir, exist_ok=True)
        output_name = f"{output_dir}/{filename}"
        df_processed.to_csv(output_name, index=False)
        #print(ouput_name)
        #plot_phasic_and_tonic(df_processed)

    result_df = pd.DataFrame(results)
    output_name = "Meritve/Filtered/cvxeda/11eda_feature_summary_10hz.csv"
    result_df.to_csv(output_name, index=False)
    print(f"""
          average correlation: {result_df['Correlation'].mean():.4f}
          average tonic correlation: {result_df['Tonic_Correlation'].mean():.4f}
          average phasic correlation: {result_df['Phasic_Correlation'].mean():.4f}
          """)

if __name__ == "__main__":
    main()