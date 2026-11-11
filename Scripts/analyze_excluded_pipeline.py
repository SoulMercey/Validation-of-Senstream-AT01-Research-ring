"""Run the analysis pipeline for the six excluded participant recordings.

The extra ``P018 2 movement test_combined.csv`` recording is intentionally
omitted.  Results and freshly filtered signals are written below
``Results/excluded_pipeline`` so the main analysis outputs are not changed.
"""

from pathlib import Path

import pandas as pd

from compare_peak_detection_scores import get_phasic_peaks, match_peaks
from cvxeda_filtering_export import process_eda_cvx
from file_exclusions import load_excluded_files
from manual_filtering_export import process_eda_manual
from neurokit_filtering_export import make_dataframe, process_eda
from quality_metrics_filtered import compute_pair_metrics
from signal_utils import trim_initial_seconds


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "Results" / "excluded_pipeline"
COMBINED_DIR = REPO_ROOT / "Meritve" / "Combined"
TEST_RECORDING = "P018 2 movement test_combined.csv"
TOLERANCE_SAMPLES = 10  # 10 Hz: the main pipeline uses +/- 1 second


def _recordings() -> list[Path]:
    names = sorted(
        (name for name in load_excluded_files() if name != TEST_RECORDING),
        key=str.casefold,
    )
    paths = []
    for name in names:
        matches = [path for path in COMBINED_DIR.rglob("*_combined.csv") if path.name == name]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one exact match for {name!r}, found {len(matches)}")
        paths.append(matches[0])
    return paths


def _neurokit_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    senstream, _, _ = process_eda(df["senstream"].copy())
    biopac, _, _ = process_eda(df["biopac"].copy())
    return make_dataframe(biopac, senstream, df.copy())


def _metric_columns(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key.lower()}": value for key, value in metrics.items()}


def run_excluded_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered_root = OUTPUT_DIR / "filtered"
    correlation_rows = []
    peak_rows = []
    paths = _recordings()

    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}")
        source = pd.read_csv(path)
        source["time"] = pd.to_datetime(source["time"], errors="coerce")

        # Run all three decompositions afresh with the same functions and
        # parameters as run_all.py.
        manual = process_eda_manual(source.copy(), filename=path.name)
        neurokit = _neurokit_decomposition(source.copy())
        cvxeda, _, _ = process_eda_cvx(source.copy())
        decompositions = {
            "manual": manual,
            "neurokit": neurokit,
            "cvxeda": cvxeda,
        }

        relative_day = path.parent.name
        for method, frame in decompositions.items():
            destination = filtered_root / method / relative_day / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(destination, index=False)

        # Exactly as in the main evaluation, discard the 60-second
        # acclimatisation period only after decomposition.
        raw_eval = trim_initial_seconds(source)
        raw_metrics = compute_pair_metrics(raw_eval, "biopac", "senstream")
        for method, frame in decompositions.items():
            evaluated = trim_initial_seconds(frame)
            tonic = compute_pair_metrics(evaluated, "Biopac_Tonic", "Senstream_Tonic")
            phasic = compute_pair_metrics(evaluated, "Biopac_Phasic", "Senstream_Phasic")
            correlation_rows.append({
                "file": path.name,
                "method": method,
                **_metric_columns("raw", raw_metrics),
                **_metric_columns("tonic", tonic),
                **_metric_columns("phasic", phasic),
            })

        manual_eval = trim_initial_seconds(manual)
        neurokit_eval = trim_initial_seconds(neurokit)
        cvxeda_eval = trim_initial_seconds(cvxeda)
        peak_sets = get_phasic_peaks(manual_eval, neurokit_eval, cvxeda_eval)
        named_peak_sets = zip(
            (
                "manual",
                "neurokit",
                "cvxeda",
                "prefiltered_manual",
                "prefiltered_neurokit",
                "prefiltered_cvxeda",
            ),
            peak_sets,
        )
        for method, peaks in named_peak_sets:
            bio_count = len(peaks["Biopac"]["SCR_Peaks"])
            sen_count = len(peaks["Senstream"]["SCR_Peaks"])
            _, _, scores = match_peaks(
                peaks,
                signal_length=len(manual_eval),
                tolerance_window_samples=TOLERANCE_SAMPLES,
            )
            matched = int(round(scores["recall"] * bio_count))
            peak_rows.append({
                "file": path.name,
                "method": method,
                "biopac_peaks": bio_count,
                "senstream_peaks": sen_count,
                "matched_peaks": matched,
                **scores,
            })

    correlations = pd.DataFrame(correlation_rows)
    peaks = pd.DataFrame(peak_rows)
    correlations.to_csv(OUTPUT_DIR / "correlations.csv", index=False)
    peaks.to_csv(OUTPUT_DIR / "peak_detection.csv", index=False)

    corr_summary = correlations.groupby("method", as_index=False).mean(numeric_only=True)
    peak_summary = peaks.groupby("method", as_index=False).mean(numeric_only=True)
    corr_summary.to_csv(OUTPUT_DIR / "correlations_summary_by_method.csv", index=False)
    peak_summary.to_csv(OUTPUT_DIR / "peak_detection_summary_by_method.csv", index=False)

    # A compact table with the metrics requested most often during review.
    compact_corr = correlations[[
        "file", "method", "raw_spearman", "tonic_spearman", "phasic_spearman"
    ]]
    compact_peak = peaks[[
        "file", "method", "f1_score", "peak_count_error", "avg_timing_error"
    ]]
    compact_corr.to_csv(OUTPUT_DIR / "requested_correlations.csv", index=False)
    compact_peak.to_csv(OUTPUT_DIR / "requested_peak_metrics.csv", index=False)

    print(f"Results written to {OUTPUT_DIR}")
    return correlations, peaks


if __name__ == "__main__":
    run_excluded_pipeline()
