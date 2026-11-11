"""Presteje meritve po kategorijah kakovosti za surove korelacije.

Kategorije in pragove ponovno uporabimo iz correlation_by_parts.categorize(),
da so enaki kot povsod drugod v analizi (>=0.9, >=0.7, >=0.3, sicer Weak).

Izhod: Results/Summary/raw_quality_categories.csv s stolpci
segment, metric, great, strong, medium, weak, n.
"""

import pandas as pd
from pathlib import Path

from correlation_by_parts import categorize

CATEGORY_ORDER = ["Great", "Strong", "Medium", "Weak"]

# segment -> per-file CSV v mapi Results/
SEGMENT_FILES = {
    "all": "raw_correlation_all.csv",
    "static": "raw_correlation_static.csv",
    "break": "raw_correlation_break.csv",
    "dynamic": "raw_correlation_dynamic.csv",
    "sitting": "raw_correlation_sitting.csv",
    "static_sitting": "sitting_raw_correlation_static.csv",
    "break_sitting": "sitting_raw_correlation_break.csv",
    "dynamic_sitting": "sitting_raw_correlation_dynamic.csv",
    "standing": "raw_correlation_standing.csv",
    "static_standing": "standing_raw_correlation_static.csv",
    "break_standing": "standing_raw_correlation_break.csv",
    "dynamic_standing": "standing_raw_correlation_dynamic.csv",
}

METRICS = {
    "standard_spearman": "standard_spearman",
    "max_spearman": "max_spearman",
}


def count_categories(series):
    counts = dict.fromkeys(CATEGORY_ORDER, 0)
    for value in series.dropna():
        counts[categorize(value)] += 1
    return counts


def main():
    script_path = Path(__file__).resolve()
    results_dir = script_path.parent.parent / "Results"
    output_dir = results_dir / "Summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for segment, file_name in SEGMENT_FILES.items():
        df = pd.read_csv(results_dir / file_name)
        for metric_name, column in METRICS.items():
            counts = count_categories(df[column])
            rows.append(
                {
                    "segment": segment,
                    "metric": metric_name,
                    "great": counts["Great"],
                    "strong": counts["Strong"],
                    "medium": counts["Medium"],
                    "weak": counts["Weak"],
                    "n": sum(counts.values()),
                }
            )

    out_df = pd.DataFrame(rows)
    output_path = output_dir / "raw_quality_categories.csv"
    out_df.to_csv(output_path, index=False)
    print(out_df.to_string(index=False))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
