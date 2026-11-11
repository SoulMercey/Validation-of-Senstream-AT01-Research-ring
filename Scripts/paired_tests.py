"""Parni preizkusi znacilnosti razlik, ki jih naloga trdi v besedilu.

Vse primerjave so parne po meritvi (N = 85), zato uporabimo Wilcoxonov preizkus
predznacenih rangov. Enota preizkusa je meritev in ne posamezen vzorec signala,
zato tu ni tezave z avtokorelacijo, ki velja za p-vrednosti Spearmanovih
korelacij v tabelah rezultatov.

Preizkusi so razdeleni na tri druzine:
  segment  - ali se metrika detekcije vrhov razlikuje med segmenti meritve
             (npr. padec mere F1 od prve k drugi nalogi), locno za vsako metodo
  metoda   - ali se filtrirne metode med seboj razlikujejo po metrikah ujemanja
             na celotni meritvi (rocno / NeuroKit2 / cvxEDA)
  aplikacija - ali se lastna obdelava razlikuje od obdelave v aplikaciji
             Senstream po metrikah detekcije vrhov

Izhod: Results/Summary/paired_tests.csv s stolpci
family, comparison, metric, segment, n, mean_a, mean_b, mean_diff, median_diff,
n_a_gt_b, n_b_gt_a, n_tie, wilcoxon_stat, p_value.

mean_diff je vedno mean_b - mean_a, torej sprememba od prvega k drugemu clenu
primerjave.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

# Metode dekompozicije, kot so poimenovane v Results/peak_detection*.csv
METHODS = ["manual", "neurokit", "cvxeda"]

# Metoda -> datoteka z metrikami ujemanja filtriranih komponent na celotni meritvi
METHOD_QUALITY_FILES = {
    "manual": "manual_quality_metrics.csv",
    "neurokit": "nk2_quality_metrics.csv",
    "cvxeda": "cvxeda_quality_metrics.csv",
}

# Pari segmentov za druzino "segment". Segmenti so imenovani kot v
# peak_detection_scores_segmented.csv: static = 1. naloga, break = odmor,
# dynamic = 2. naloga.
SEGMENT_PAIRS = [
    ("static", "dynamic"),
    ("static", "break"),
    ("break", "dynamic"),
]

PEAK_METRICS = ["precision", "recall", "f1_score", "peak_count_error"]

# Metrike ujemanja komponent, ki jih naloga primerja med metodami
QUALITY_METRICS = [
    "Tonic_Spearman",
    "Tonic_Max_Spearman",
    "Tonic_MAE",
    "Tonic_RMSE",
    "Phasic_Spearman",
    "Phasic_Max_Spearman",
    "Phasic_MAE",
    "Phasic_RMSE",
]

METHOD_PAIRS = [
    ("manual", "neurokit"),
    ("neurokit", "cvxeda"),
    ("manual", "cvxeda"),
]


def base_name(file_name):
    """Iz imena datoteke naredi kljuc meritve.

    'P001 1_combined_static.csv' -> 'P001 1'
    'P001 1_combined.csv'        -> 'P001 1'
    """
    return str(file_name).split("_combined")[0]


def paired_test(a, b):
    """Wilcoxonov preizkus na parnih vzorcih a in b.

    Vrne slovar z opisno statistiko in p-vrednostjo. Kadar so vse razlike
    natanko nic, preizkus ni definiran; v tem primeru vrnemo p = 1,0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a

    result = {
        "n": len(diff),
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "mean_diff": diff.mean(),
        "median_diff": np.median(diff),
        "n_a_gt_b": int((diff < 0).sum()),
        "n_b_gt_a": int((diff > 0).sum()),
        "n_tie": int((diff == 0).sum()),
    }

    if np.all(diff == 0):
        result["wilcoxon_stat"] = 0.0
        result["p_value"] = 1.0
    else:
        stat, pval = wilcoxon(a, b)
        result["wilcoxon_stat"] = float(stat)
        result["p_value"] = float(pval)

    return result


def segment_tests(results_dir):
    """Primerja metrike detekcije vrhov med segmenti, locno za vsako metodo."""
    df = pd.read_csv(results_dir / "peak_detection_scores_segmented.csv")
    df["base"] = df["File"].map(base_name)

    rows = []
    for method in METHODS:
        sub = df[df["method"] == method]
        for metric in PEAK_METRICS:
            table = sub.pivot_table(index="base", columns="Segment", values=metric)
            for seg_a, seg_b in SEGMENT_PAIRS:
                if not {seg_a, seg_b}.issubset(table.columns):
                    continue
                pair = table[[seg_a, seg_b]].dropna()
                row = {
                    "family": "segment",
                    "comparison": f"{method}: {seg_a} -> {seg_b}",
                    "metric": metric,
                    "segment": f"{seg_a}|{seg_b}",
                }
                row.update(paired_test(pair[seg_a], pair[seg_b]))
                rows.append(row)
    return rows


def method_tests(results_dir):
    """Primerja filtrirne metode po metrikah ujemanja na celotni meritvi."""
    frames = {}
    for method, file_name in METHOD_QUALITY_FILES.items():
        frame = pd.read_csv(results_dir / file_name)
        frame["base"] = frame["File"].map(base_name)
        frames[method] = frame.set_index("base")

    rows = []
    for method_a, method_b in METHOD_PAIRS:
        frame_a = frames[method_a]
        frame_b = frames[method_b]
        common = frame_a.index.intersection(frame_b.index)
        for metric in QUALITY_METRICS:
            if metric not in frame_a.columns or metric not in frame_b.columns:
                continue
            pair = pd.DataFrame(
                {
                    "a": frame_a.loc[common, metric],
                    "b": frame_b.loc[common, metric],
                }
            ).dropna()
            row = {
                "family": "metoda",
                "comparison": f"{method_a} -> {method_b}",
                "metric": metric,
                "segment": "all",
            }
            row.update(paired_test(pair["a"], pair["b"]))
            rows.append(row)
    return rows


def app_tests(results_dir):
    """Primerja lastno obdelavo z obdelavo v aplikaciji Senstream.

    V peak_detection_all.csv so vrstice 'prefiltered_<metoda>' rezultat detekcije
    na signalu, ki ga filtrira aplikacija; '<metoda>' pa na lastni obdelavi.
    """
    df = pd.read_csv(results_dir / "peak_detection_all.csv")
    df["base"] = df["file"].map(base_name)

    rows = []
    for method in METHODS:
        own = df[df["method"] == method].set_index("base")
        app = df[df["method"] == f"prefiltered_{method}"].set_index("base")
        common = own.index.intersection(app.index)
        for metric in PEAK_METRICS:
            pair = pd.DataFrame(
                {"a": own.loc[common, metric], "b": app.loc[common, metric]}
            ).dropna()
            row = {
                "family": "aplikacija",
                "comparison": f"{method} -> aplikacija ({method})",
                "metric": metric,
                "segment": "all",
            }
            row.update(paired_test(pair["a"], pair["b"]))
            rows.append(row)
    return rows


def main():
    script_path = Path(__file__).resolve()
    results_dir = script_path.parent.parent / "Results"
    output_dir = results_dir / "Summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(segment_tests(results_dir))
    rows.extend(method_tests(results_dir))
    rows.extend(app_tests(results_dir))

    columns = [
        "family",
        "comparison",
        "metric",
        "segment",
        "n",
        "mean_a",
        "mean_b",
        "mean_diff",
        "median_diff",
        "n_a_gt_b",
        "n_b_gt_a",
        "n_tie",
        "wilcoxon_stat",
        "p_value",
    ]
    out_df = pd.DataFrame(rows)[columns]

    output_path = output_dir / "paired_tests.csv"
    out_df.to_csv(output_path, index=False)

    # Na konzolo izpisemo le tiste primerjave, ki jih naloga navaja v besedilu.
    highlights = out_df[
        (
            (out_df["family"] == "segment")
            & (out_df["metric"] == "f1_score")
            & (out_df["segment"] == "static|dynamic")
        )
        | (
            (out_df["family"] == "metoda")
            & (out_df["comparison"].isin(["neurokit -> cvxeda", "manual -> neurokit"]))
            & (out_df["metric"].isin(["Tonic_Spearman", "Phasic_Spearman", "Phasic_MAE"]))
        )
        | ((out_df["family"] == "aplikacija") & (out_df["metric"] == "f1_score"))
    ]

    display = highlights.copy()
    for column in ["mean_a", "mean_b", "mean_diff", "median_diff"]:
        display[column] = display[column].round(3)
    display["p_value"] = display["p_value"].map(lambda value: f"{value:.2g}")

    print(display.to_string(index=False))
    print(f"\n{len(out_df)} preizkusov skupaj, shranjeno v {output_path}")


if __name__ == "__main__":
    main()
