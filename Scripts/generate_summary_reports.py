import pandas as pd
import os
from pathlib import Path


results_dir = Path("Results")
output_dir = Path("Results/Summary")

def get_segments(df):
    df_static = df[df["Segment"] == "static"]
    df_dynamic = df[df["Segment"] == "dynamic"]
    df_break = df[df["Segment"] == "break"]
    return df_static, df_dynamic, df_break

#basically fora je da to k mam za vsak file posebi napisane metrike je kinda neuporabno. Tukile bom naredu za vsako metodo skupen file 
def get_means(df, segment=None):
    if segment:
        df = df[df["segment"] == segment]
    smna_cols = [col for col in df.columns if "SMNA" in col] # remove smna columns from cvxeda
    if smna_cols:
        df = df.drop(columns=smna_cols)
    means = df.mean(numeric_only=True)
    return means.to_numpy()

def get_peak_means(df, segment=None):
    means = []
    if segment == "static" or segment == "dynamic" or segment == "break":
        df = df[df["segment"] == segment]
    #exctract only one method at a time
    df_manual = df[df["method"] == "manual"]
    df_nk = df[df["method"] == "neurokit"]
    df_cvx = df[df["method"] == "cvxeda"]
    means.append(["manual", *get_means(df_manual, segment)])
    means.append(["neurokit", *get_means(df_nk, segment)])
    means.append(["cvxeda", *get_means(df_cvx, segment)])
    return means


# --- razprsenost za filtrirane metrike in detekcijo vrhov -------------------
# Enako nacelo kot pri get_raw_stats(): poleg povprecij porocamo se standardne
# odklone, N in stevilo meritev s p < 0.001. Povprecje p-vrednosti ni veljavna
# statistika, zato stolpce *_pval_mean ohranimo le zaradi zdruzljivosti.

_COMPONENTS = ("Tonic", "Phasic")
_BASE_METRICS = [
    "Spearman", "Spearman_pval", "Max_Spearman", "Max_Spearman_pval",
    "Lag_Seconds", "MAE", "RMSE", "NRMSE_Range", "NRMSE_STD",
    "BA_Bias", "BA_Lower_LOA", "BA_Upper_LOA",
]

FILTERED_SUMMARY_COLS = (
    ["method", "segment"]
    + [f"{c}_{m}_mean" for c in _COMPONENTS for m in _BASE_METRICS]
    + ["n_files"]
    + [
        f"{c}_{m}"
        for c in _COMPONENTS
        for m in (
            "Spearman_std", "Max_Spearman_std", "MAE_std", "RMSE_std",
            "BA_Bias_std", "LOA_Width_mean", "LOA_Width_std",
            "Lag_Abs_mean", "Lag_Abs_std",
            "Spearman_n_p001", "Max_Spearman_n_p001",
        )
    ]
)


def get_filtered_stats(df):
    """Povprecja (v obstojecem vrstnem redu) + razprsenost, N in stevci znacilnosti."""
    values = [df[f"{c}_{m}"].mean() for c in _COMPONENTS for m in _BASE_METRICS]
    values.append(len(df))
    for c in _COMPONENTS:
        lag = df[f"{c}_Lag_Seconds"]
        loa_width = df[f"{c}_BA_Upper_LOA"] - df[f"{c}_BA_Lower_LOA"]
        values += [
            df[f"{c}_Spearman"].std(ddof=1),
            df[f"{c}_Max_Spearman"].std(ddof=1),
            df[f"{c}_MAE"].std(ddof=1),
            df[f"{c}_RMSE"].std(ddof=1),
            df[f"{c}_BA_Bias"].std(ddof=1),
            loa_width.mean(),
            loa_width.std(ddof=1),
            lag.abs().mean(),
            lag.abs().std(ddof=1),
            int((df[f"{c}_Spearman_pval"] < 0.001).sum()),
            int((df[f"{c}_Max_Spearman_pval"] < 0.001).sum()),
        ]
    return values


# Poleg treh metod dekompozicije porocamo se detekcijo vrhov na signalu, ki ga
# filtrira aplikacija Senstream (prefiltered_*).
PEAK_METHODS = [
    "manual", "neurokit", "cvxeda",
    "prefiltered_manual", "prefiltered_neurokit", "prefiltered_cvxeda",
]
_PEAK_METRICS = ["precision", "recall", "f1_score", "peak_count_error", "avg_timing_error"]

PEAK_SUMMARY_COLS = (
    ["segment", "method"]
    + [f"{m}_mean" for m in _PEAK_METRICS]
    + ["n_files"]
    + [f"{m}_std" for m in _PEAK_METRICS]
    + ["n_timing"]
)


def get_peak_stats(df):
    """Za vsako metodo vrne [method, povprecja..., N, odkloni..., n_timing].

    avg_timing_error je NaN pri meritvah brez ujemanj, zato uporabimo mean/std,
    ki NaN preskocita, in posebej porocamo n_timing = stevilo meritev z ujemanjem.
    """
    rows = []
    for method in PEAK_METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        row = [method]
        row += [sub[m].mean() for m in _PEAK_METRICS]
        row.append(len(sub))
        row += [sub[m].std(ddof=1) for m in _PEAK_METRICS]
        row.append(int(sub["avg_timing_error"].notna().sum()))
        rows.append(row)
    return rows

RAW_SUMMARY_COLS = [
    "segment",
    # obstojeca povprecja - imen ne spreminjaj, bere jih analyze_results.py
    "standard_spearman_mean",
    "standard_spearman_pval_mean",
    "max_spearman_mean",
    "max_spearman_pval_mean",
    "lag_seconds_mean",
    "raw_nrmse_range_mean",
    "raw_nrmse_std_mean",
    # razprsenost, N in stevci znacilnosti za tabele v nalogi
    "n_files",
    "standard_spearman_std",
    "max_spearman_std",
    "lag_seconds_std",
    "lag_seconds_abs_mean",
    "lag_seconds_abs_std",
    "standard_spearman_n_p001",
    "max_spearman_n_p001",
]


def get_raw_stats(df):
    """Povprecja, standardni odkloni, N in stevilo meritev s p < 0.001.

    Povprecje p-vrednosti ni veljavna statistika, zato poleg njega porocamo se
    stevilo meritev s p < 0.001. Signirano povprecje zamika se lahko iznici,
    zato dodamo se povprecje absolutnih zamikov.
    """
    lag = df["lag_seconds"]
    return [
        df["standard_spearman"].mean(),
        df["standard_spearman_pval"].mean(),
        df["max_spearman"].mean(),
        df["max_spearman_pval"].mean(),
        lag.mean(),
        df["raw_nrmse_range"].mean(),
        df["raw_nrmse_std"].mean(),
        len(df),
        df["standard_spearman"].std(ddof=1),
        df["max_spearman"].std(ddof=1),
        lag.std(ddof=1),
        lag.abs().mean(),
        lag.abs().std(ddof=1),
        int((df["standard_spearman_pval"] < 0.001).sum()),
        int((df["max_spearman_pval"] < 0.001).sum()),
    ]


def summarize_raw():
    df_all = pd.read_csv("Results/raw_correlation_all.csv")
    df_sitting = pd.read_csv("Results/raw_correlation_sitting.csv")
    df_standing = pd.read_csv("Results/raw_correlation_standing.csv")
    df_dynamic = pd.read_csv("Results/raw_correlation_dynamic.csv")
    df_break = pd.read_csv("Results/raw_correlation_break.csv")
    df_static = pd.read_csv("Results/raw_correlation_static.csv")
    df_static_sitting = pd.read_csv("Results/sitting_raw_correlation_static.csv")
    df_dynamic_sitting = pd.read_csv("Results/sitting_raw_correlation_dynamic.csv")
    df_break_sitting = pd.read_csv("Results/sitting_raw_correlation_break.csv")
    df_static_standing = pd.read_csv("Results/standing_raw_correlation_static.csv")
    df_dynamic_standing = pd.read_csv("Results/standing_raw_correlation_dynamic.csv")
    df_break_standing = pd.read_csv("Results/standing_raw_correlation_break.csv")
    rows = []
    summary_cols = RAW_SUMMARY_COLS
    rows.append(["all", *get_raw_stats(df_all)])
    rows.append(["sitting", *get_raw_stats(df_sitting)])
    rows.append(["standing", *get_raw_stats(df_standing)])
    rows.append(["static", *get_raw_stats(df_static)])
    rows.append(["dynamic", *get_raw_stats(df_dynamic)])
    rows.append(["break", *get_raw_stats(df_break)])
    rows.append(["static_sitting", *get_raw_stats(df_static_sitting)])
    rows.append(["dynamic_sitting", *get_raw_stats(df_dynamic_sitting)])
    rows.append(["break_sitting", *get_raw_stats(df_break_sitting)])
    rows.append(["static_standing", *get_raw_stats(df_static_standing)])
    rows.append(["dynamic_standing", *get_raw_stats(df_dynamic_standing)])
    rows.append(["break_standing", *get_raw_stats(df_break_standing)])
    summary_df = pd.DataFrame(rows, columns=summary_cols)

    #print(summary_df.to_string(index=False))
    summary_df.to_csv(output_dir / "raw_summary.csv", index=False)


def summarize_nk2():
    df_all = pd.read_csv("Results/nk2_quality_metrics.csv")
    df_sitting = pd.read_csv("Results/nk2_quality_metrics_sitting.csv")
    df_standing = pd.read_csv("Results/nk2_quality_metrics_standing.csv")
    df_segmented = pd.read_csv("Results/nk2_quality_metrics_segmented.csv")
    df_segmented_sitting = pd.read_csv("Results/nk2_quality_metrics_segmented_sitting.csv")
    df_segmented_standing = pd.read_csv("Results/nk2_quality_metrics_segmented_standing.csv")
    df_static, df_dynamic, df_break = get_segments(df_segmented)
    df_static_sitting, df_dynamic_sitting, df_break_sitting = get_segments(df_segmented_sitting)
    df_static_standing, df_dynamic_standing, df_break_standing = get_segments(df_segmented_standing)
    cols = FILTERED_SUMMARY_COLS
    
    rows = []
    rows.append(["neurokit","all", *get_filtered_stats(df_all)])
    rows.append(["neurokit","sitting", *get_filtered_stats(df_sitting)])
    rows.append(["neurokit","standing", *get_filtered_stats(df_standing)])
    rows.append(["neurokit","static", *get_filtered_stats(df_static)])
    rows.append(["neurokit","dynamic", *get_filtered_stats(df_dynamic)])
    rows.append(["neurokit","break", *get_filtered_stats(df_break)])
    rows.append(["neurokit","static_sitting", *get_filtered_stats(df_static_sitting)])
    rows.append(["neurokit","dynamic_sitting", *get_filtered_stats(df_dynamic_sitting)])
    rows.append(["neurokit","break_sitting", *get_filtered_stats(df_break_sitting)])
    rows.append(["neurokit","static_standing", *get_filtered_stats(df_static_standing)])
    rows.append(["neurokit","dynamic_standing", *get_filtered_stats(df_dynamic_standing)])
    rows.append(["neurokit","break_standing", *get_filtered_stats(df_break_standing)])
    summary_df = pd.DataFrame(rows, columns=cols)
    #print(summary_df.to_string(index=False))
    #summary_df.to_csv(output_dir / "nk2_summary.csv", index=False)
    return summary_df

def summarize_cvx():
    df_all = pd.read_csv("Results/cvxeda_quality_metrics.csv")
    df_sitting = pd.read_csv("Results/cvxeda_quality_metrics_sitting.csv")
    df_standing = pd.read_csv("Results/cvxeda_quality_metrics_standing.csv")
    df_segmented = pd.read_csv("Results/cvxeda_quality_metrics_segmented.csv")
    df_segmented_sitting = pd.read_csv("Results/cvxeda_quality_metrics_segmented_sitting.csv")
    df_segmented_standing = pd.read_csv("Results/cvxeda_quality_metrics_segmented_standing.csv")
    df_static, df_dynamic, df_break = get_segments(df_segmented)
    df_static_sitting, df_dynamic_sitting, df_break_sitting = get_segments(df_segmented_sitting)
    df_static_standing, df_dynamic_standing, df_break_standing = get_segments(df_segmented_standing)
    cols = FILTERED_SUMMARY_COLS
    
    rows = []
    rows.append(["cvxeda","all", *get_filtered_stats(df_all)])
    rows.append(["cvxeda","sitting", *get_filtered_stats(df_sitting)])
    rows.append(["cvxeda","standing", *get_filtered_stats(df_standing)])
    rows.append(["cvxeda","static", *get_filtered_stats(df_static)])
    rows.append(["cvxeda","dynamic", *get_filtered_stats(df_dynamic)])
    rows.append(["cvxeda","break", *get_filtered_stats(df_break)])
    rows.append(["cvxeda","static_sitting", *get_filtered_stats(df_static_sitting)])
    rows.append(["cvxeda","dynamic_sitting", *get_filtered_stats(df_dynamic_sitting)])
    rows.append(["cvxeda","break_sitting", *get_filtered_stats(df_break_sitting)])
    rows.append(["cvxeda","static_standing", *get_filtered_stats(df_static_standing)])
    rows.append(["cvxeda","dynamic_standing", *get_filtered_stats(df_dynamic_standing)])
    rows.append(["cvxeda","break_standing", *get_filtered_stats(df_break_standing)])
    summary_df = pd.DataFrame(rows, columns=cols)
    #print(summary_df.to_string(index=False))
    #summary_df.to_csv(output_dir / "cvxeda_summary.csv", index=False)
    return summary_df

def summarize_manual():
    df_all = pd.read_csv("Results/manual_quality_metrics.csv")
    df_sitting = pd.read_csv("Results/manual_quality_metrics_sitting.csv")
    df_standing = pd.read_csv("Results/manual_quality_metrics_standing.csv")
    df_segmented = pd.read_csv("Results/manual_quality_metrics_segmented.csv")
    df_segmented_sitting = pd.read_csv("Results/manual_quality_metrics_segmented_sitting.csv")
    df_segmented_standing = pd.read_csv("Results/manual_quality_metrics_segmented_standing.csv")
    df_static, df_dynamic, df_break = get_segments(df_segmented)
    df_static_sitting, df_dynamic_sitting, df_break_sitting = get_segments(df_segmented_sitting)
    df_static_standing, df_dynamic_standing, df_break_standing = get_segments(df_segmented_standing)
    cols = FILTERED_SUMMARY_COLS
    
    rows = []
    rows.append(["manual","all", *get_filtered_stats(df_all)])
    rows.append(["manual","sitting", *get_filtered_stats(df_sitting)])
    rows.append(["manual","standing", *get_filtered_stats(df_standing)])
    rows.append(["manual","static", *get_filtered_stats(df_static)])
    rows.append(["manual","dynamic", *get_filtered_stats(df_dynamic)])
    rows.append(["manual","break", *get_filtered_stats(df_break)])
    rows.append(["manual","static_sitting", *get_filtered_stats(df_static_sitting)])
    rows.append(["manual","dynamic_sitting", *get_filtered_stats(df_dynamic_sitting)])
    rows.append(["manual","break_sitting", *get_filtered_stats(df_break_sitting)])
    rows.append(["manual","static_standing", *get_filtered_stats(df_static_standing)])
    rows.append(["manual","dynamic_standing", *get_filtered_stats(df_dynamic_standing)])
    rows.append(["manual","break_standing", *get_filtered_stats(df_break_standing)])
    summary_df = pd.DataFrame(rows, columns=cols)
    #print(summary_df.to_string(index=False))
    #summary_df.to_csv(output_dir / "manual_summary.csv", index=False)
    return summary_df

def generate_peak_list(df,segment):
    rows = []
    for row in get_peak_stats(df):
        rows.append([segment, *row])
    return rows

def summarize_peaks():
    """
    This is already made in compare_peak_detection_scores.py at end of main
    """
    df_all = pd.read_csv("Results/peak_detection_all.csv")
    # sitting/standing morata biti povprecji cez CELOTNE meritve, ne cez segmente,
    # sicer nista primerljivi z vrstico "all" niti z raw_summary.csv
    df_sitting = pd.read_csv("Results/peak_detection_sitting.csv")
    df_standing = pd.read_csv("Results/peak_detection_standing.csv")
    df_segmented = pd.read_csv("Results/peak_detection_scores_segmented.csv")
    df_segmented_sitting = pd.read_csv("Results/peak_detection_scores_segmented_sitting.csv")
    df_segmented_standing = pd.read_csv("Results/peak_detection_scores_segmented_standing.csv")
    df_static, df_dynamic, df_break = get_segments(df_segmented)
    df_static_sitting, df_dynamic_sitting, df_break_sitting = get_segments(df_segmented_sitting)
    df_static_standing, df_dynamic_standing, df_break_standing = get_segments(df_segmented_standing)
    rows = []
    summary_cols = PEAK_SUMMARY_COLS
    rows += generate_peak_list(df_all,"all")
    rows += generate_peak_list(df_sitting,"sitting")
    rows += generate_peak_list(df_standing,"standing")
    rows += generate_peak_list(df_static, "static")
    rows += generate_peak_list(df_dynamic, "dynamic")
    rows += generate_peak_list(df_break, "break")
    rows += generate_peak_list(df_static_sitting, "static_sitting")
    rows += generate_peak_list(df_dynamic_sitting, "dynamic_sitting")
    rows += generate_peak_list(df_break_sitting, "break_sitting")
    rows += generate_peak_list(df_static_standing, "static_standing")
    rows += generate_peak_list(df_dynamic_standing, "dynamic_standing")
    rows += generate_peak_list(df_break_standing, "break_standing")



    summary_df = pd.DataFrame(rows, columns=summary_cols)
    summary_df.to_csv(output_dir / "peak_summary.csv", index=False)
    
def summarize_filtered():
    b = summarize_nk2()
    c = summarize_cvx()
    a = summarize_manual()
    df = pd.concat([a,b,c])
    df.to_csv(output_dir/"filtered_summary.csv",index=False)

    



def main():
    os.makedirs(output_dir, exist_ok=True)
    summarize_raw()
    summarize_filtered()
    summarize_peaks()

if __name__ == "__main__":
    main()