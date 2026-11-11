"""Ustvari histograma EDA-vrednosti čez vse vključene meritve."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from file_exclusions import filter_csv_paths


ROOT_DIR = Path(__file__).resolve().parent.parent
COMBINED_DIR = ROOT_DIR / "Meritve" / "Combined"
OUTPUT_DIR = ROOT_DIR / "thesis" / "Slike"


def decimal_comma(value, _position):
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", ",")


def main(file_path=None):
    """Ustvari histograma za vse meritve ali za podano datoteko.

    Privzeto se uporabijo vse kombinirane meritve, razen datotek iz
    ``Scripts/file_exclusions.csv``. Argument ``file_path`` ostane na voljo
    za izdelavo histograma posamezne meritve.
    """
    if file_path is None:
        paths = sorted(COMBINED_DIR.rglob("*_combined.csv"))
        paths = filter_csv_paths(paths)
    else:
        paths = [Path(file_path)]

    if not paths:
        raise FileNotFoundError(f"V mapi {COMBINED_DIR} ni najdenih meritev.")

    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["senstream", "biopac"])
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    histograms = [
        ("senstream", "Senstream AT01", "red", "senstream_histogram.pdf"),
        ("biopac", "Biopac", "blue", "biopac_histogram.pdf"),
    ]

    for column, _title, color, filename in histograms:
        values = data[column].dropna()
        fig, ax = plt.subplots()
        ax.hist(values, bins=30, color=color, edgecolor="black")
        ax.set_xlabel(r"Vrednost ($\mu$S)")
        ax.set_ylabel("Frekvenca")
        formatter = FuncFormatter(decimal_comma)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        fig.savefig(OUTPUT_DIR / filename, format="pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Histogrami ustvarjeni iz {len(paths)} meritev v mapi {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
