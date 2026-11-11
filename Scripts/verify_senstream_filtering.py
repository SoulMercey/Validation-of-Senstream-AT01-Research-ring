"""Preveri, ali je stolpec `senstream_filtered` res fazicni signal.

Aplikacija Senstream poleg surovega signala shrani tudi svojo filtrirano razlicico.
V analizi jo obravnavamo kot fazicno komponento, kar je predpostavka in ne dejstvo.
Ta skripta jo preveri na dva nacina:

  1. Povprecni signal cez vse meritve. Ce gre res za fazicno komponento, se mora
     povprecje obnasati enako kot povprecji nasih dveh fazicnih komponent, torej
     imeti stiri vrhove ob hiperventilacijah in biti drugod ravno.
  2. Korelacije s fazicnimi komponentami obeh naprav pri vseh treh metodah
     dekompozicije. Za primerjavo izpisemo se korelacijo s surovim signalom
     Senstreama; ce je ta visja od korelacij s fazicnimi komponentami, filtrirani
     signal ni fazicna komponenta, temvec le zglajen surov signal.

Skripta nicesar ne shranjuje. Graf le prikaze, stevilke izpise v konzolo.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_results import (average_phasic, apply_decimal_comma, COLOR_BIOPAC,
                             COLOR_SENSTREAM, COLOR_INK, COLOR_MUTED)
from file_exclusions import filter_csv_paths

FS = 10
ACCLIMATIZATION_S = 60          # prvih 60 s je aklimatizacija in jo povsod izpustimo
COLOR_FILTERED = "#000000"      # tretja barva; locljivost pri barvni slepoti preverjena

METHODS = {"manual": "ročno", "neurokit2": "NeuroKit2", "cvxeda": "cvxEDA"}
FILTERED_COLUMN = "senstream_filtered"


def _peaks(grand, t, threshold=1.0, merge_s=10):
    """Lokalni maksimumi nad pragom, zdruzeni ce so blizu skupaj."""
    idx = [i for i in range(1, len(grand) - 1)
           if grand[i] > threshold and grand[i] >= grand[i - 1] and grand[i] > grand[i + 1]]
    merged = []
    for i in idx:
        if not merged or t[i] - t[merged[-1]] > merge_s:
            merged.append(i)
        elif grand[i] > grand[merged[-1]]:
            merged[-1] = i
    return merged


def plot_average_with_filtered(output_name=None):
    """Povprecne krivulje obeh nasih fazicnih komponent in filtriranega signala.

    Enak graf kot plot_average_phasic(), le da je dodana tretja krivulja.
    Brez `output_name` graf le prikaze, z njim ga shrani v thesis/Slike.
    """
    series = [("Biopac_Phasic", "Biopac, fazična", COLOR_BIOPAC),
              ("Senstream_Phasic", "Senstream, fazična", COLOR_SENSTREAM),
              (FILTERED_COLUMN, "Senstream, filtriran v aplikaciji", COLOR_FILTERED)]

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axhline(0, color=COLOR_MUTED, lw=0.8, zorder=2)

    for column, name, color in series:
        grand, n_files = average_phasic(column)
        t = np.arange(len(grand)) / FS
        ax.plot(t, grand, color=color, lw=1.7, zorder=3, label=name)
        found = _peaks(grand, t)
        print(f"{name:34s} vrhov nad 1,0 SD: {len(found)}"
              + ("   pri t = " + ",  ".join(f"{t[i]:.1f} s ({grand[i]:.2f})" for i in found)
                 if found else ""))

    ax.set_xlabel("Čas (s)", fontsize=11, color=COLOR_INK)
    ax.set_ylabel("Povprečen signal (SD)", fontsize=11, color=COLOR_INK)
    ax.set_xlim(0, t[-1])
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.grid(color="#e6e6e2", lw=0.6, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d7d7d2")
    ax.tick_params(colors=COLOR_MUTED, labelsize=9.5)
    apply_decimal_comma(fig)
    plt.tight_layout()
    if output_name is None:
        ax.set_title(f"Povprečje {n_files} meritev", fontsize=11, color=COLOR_INK)
        plt.show()
        return None

    # naslov izpustimo, ker ga v nalogi nosi napis pod sliko
    output = Path(__file__).resolve().parent.parent / "thesis" / "Slike" / f"{output_name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nGraf shranjen: {output}   (povprecje {n_files} meritev)")
    return output


def correlations_with_filtered():
    """Korelacije filtriranega signala s fazicnimi komponentami pri vseh metodah."""
    base = Path(__file__).resolve().parent.parent / "Meritve" / "Filtered"
    start = ACCLIMATIZATION_S * FS

    rows = []
    for method in METHODS:
        for path in filter_csv_paths(sorted((base / method).rglob("*_combined.csv"))):
            df = pd.read_csv(path).iloc[start:]
            get = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy()
            filtered = get(FILTERED_COLUMN)
            for target, label in (("Biopac_Phasic", "Biopac fazična"),
                                  ("Senstream_Phasic", "Senstream fazična"),
                                  ("senstream", "Senstream surov")):
                a, b = filtered, get(target)
                mask = ~np.isnan(a) & ~np.isnan(b)
                if mask.sum() < 3:
                    continue
                rho, _ = spearmanr(a[mask], b[mask])
                rows.append({"method": method, "target": label, "file": path.name, "rho": rho})

    df = pd.DataFrame(rows)
    n = df.file.nunique()
    print(f"\nKorelacija stolpca `{FILTERED_COLUMN}` z drugimi signali")
    print(f"Spearman po meritvah, od {ACCLIMATIZATION_S} s naprej, N = {n}\n")

    targets = ["Biopac fazična", "Senstream fazična", "Senstream surov"]
    print(f"{'metoda':12s}" + "".join(f"{t:>22s}" for t in targets))
    for method, label in METHODS.items():
        cells = []
        for target in targets:
            v = df[(df.method == method) & (df.target == target)].rho
            cells.append(f"{v.mean():.3f} +- {v.std(ddof=1):.3f}")
        print(f"{label:12s}" + "".join(f"{c:>22s}" for c in cells))

    return df

def main():
    correlations_with_filtered()
    plot_average_with_filtered("povprecna_fazicna_aplikacija")


if __name__ == "__main__":
    main()
