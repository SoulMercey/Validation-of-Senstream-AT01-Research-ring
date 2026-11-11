"""Create a visual review of every recording in file_exclusions.csv.

The plots use the same plotting function as the thesis figure
``fig:file_exclusions``.  In addition to one PDF and PNG per recording, the
script writes a single multi-page PDF and a CSV with the raw Spearman
correlation of every listed recording.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from analyze_results import (
    REPO_ROOT,
    calculate_raw_spearman_for_file,
    plot_normalized_signal_comparison,
)
from file_exclusions import load_excluded_files


OUTPUT_DIR = REPO_ROOT / "Results" / "Graphs" / "file_exclusions"


def _find_recording(filename: str) -> Path:
    combined_dir = REPO_ROOT / "Meritve" / "Combined"
    matches = [
        path for path in combined_dir.rglob("*_combined.csv")
        if path.name == filename
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one exact match for {filename!r}, found {len(matches)}"
        )
    return matches[0]


def plot_all_file_exclusions() -> Path:
    """Render all listed exclusions and return the multi-page PDF path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overview_path = OUTPUT_DIR / "file_exclusions_overview.pdf"
    rows = []

    # Sorting makes the output reproducible because the loader intentionally
    # returns a set for fast filtering elsewhere in the analysis.
    excluded = sorted(load_excluded_files(), key=str.casefold)
    with PdfPages(overview_path) as overview:
        for filename in excluded:
            path = _find_recording(filename)
            rho = calculate_raw_spearman_for_file(path)
            safe_stem = path.stem.replace("_combined", "").replace(" ", "_")
            output_stem = OUTPUT_DIR / f"signal_{safe_stem}"

            # This is the same helper used to create signal_P010_1.pdf and
            # signal_P019_1.pdf for fig:file_exclusions in the thesis.
            fig = plot_normalized_signal_comparison(path, title="")
            fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(output_stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
            overview.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            rows.append({
                "file_name": filename,
                "spearman_rho_after_60_s": rho,
                "source_path": path.relative_to(REPO_ROOT).as_posix(),
                "plot_pdf": output_stem.with_suffix(".pdf").relative_to(REPO_ROOT).as_posix(),
                "plot_png": output_stem.with_suffix(".png").relative_to(REPO_ROOT).as_posix(),
            })

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "file_exclusions_review.csv", index=False)
    print(f"Created {len(rows)} exclusion plots in {OUTPUT_DIR}")
    print(f"Overview: {overview_path}")
    return overview_path


if __name__ == "__main__":
    plot_all_file_exclusions()
