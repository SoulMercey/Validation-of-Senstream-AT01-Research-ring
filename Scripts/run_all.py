"""Run preprocessing, pairing, combination, and all available analyses."""
import sys

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from trim_biopac import main as main_trim_biopac
from trim_senstream import main as main_trim_senstream
from check_pairity import main as main_check_pairity
from make_combined import main as main_make_combined


def first_processing():
    """Regenerate trimmed raw CSVs, report pair coverage, and combine valid pairs."""
    main_trim_biopac()
    main_trim_senstream()
    main_check_pairity()
    main_make_combined()


def main():
    first_processing()

    # Import analysis packages only after raw processing has produced valid inputs.
    from calculate_correlation import main as main_calculate_correlation
    from correlation_by_parts import main as main_correlation_by_parts
    from manual_filtering_export import main as main_manual_filtering_export
    from neurokit_filtering_export import main as main_neurokit_filtering_export
    from compare_peak_detection_scores import main as main_compare_peak_detection_scores
    from compare_peak_detection_scores_by_parts import main as main_compare_peak_detection_scores_by_parts
    from quality_metrics_filtered import main as main_quality_metrics_filtered
    from quality_metrics_filtered_by_parts import main as main_quality_metrics_filtered_by_parts
    from generate_summary_reports import main as main_generate_summary_reports
    from raw_quality_categories import main as main_raw_quality_categories
    from paired_tests import main as main_paired_tests

    main_calculate_correlation()
    main_correlation_by_parts()
    try:
        from cvxeda_filtering_export import main as main_cvxeda_filtering_export
    except ImportError as exc:
        print(f"SKIP cvxEDA analysis: dependency unavailable ({exc}).")
    else:
        main_cvxeda_filtering_export()
    main_manual_filtering_export()
    main_neurokit_filtering_export()
    main_compare_peak_detection_scores()
    main_compare_peak_detection_scores_by_parts()
    main_quality_metrics_filtered()
    main_quality_metrics_filtered_by_parts()
    main_generate_summary_reports()
    main_raw_quality_categories()
    main_paired_tests()


if __name__ == "__main__":
    main()
