"""Primerja tonicno in fazicno komponento rocnega filtriranja in NeuroKit2.

Obe metodi dekompozicijo izvedeta z Butterworthovim filtrom 2. reda pri 0,05 Hz z
niclno fazo, le po razlicni poti: rocna metoda tonicno komponento dobi z
nizkoprepustnim filtrom in fazicno z odstevanjem, NeuroKit2 pa v nacinu "highpass"
(algoritem Biopac AcqKnowledge) uporabi locena filtra. Ta skripta preveri, koliko
se rezultata dejansko razlikujeta.

Primerjamo v NOTRANJOSTI signala (brez EDGE_SECONDS na vsakem robu), ker imata
scipy.signal.filtfilt in sosfiltfilt razlicno obravnavo robov in se prehodni pojav
filtra pri 0,05 Hz pozna le tam.

Izhod: Results/Summary/manual_vs_nk2.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from file_exclusions import filter_csv_paths

EDGE_SECONDS = 30
FS = 10

COMPONENTS = {
    "tonic": ("Biopac_Tonic", "Senstream_Tonic"),
    "phasic": ("Biopac_Phasic", "Senstream_Phasic"),
}


def interior(series, edge):
    return series.to_numpy()[edge:-edge]


def main():
    script_path = Path(__file__).resolve()
    base = script_path.parent.parent
    manual_dir = base / "Meritve" / "Filtered" / "manual"
    nk2_dir = base / "Meritve" / "Filtered" / "neurokit2"
    output_dir = base / "Results" / "Summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    manual_csvs = filter_csv_paths(sorted(manual_dir.rglob("*_combined.csv")))
    edge = EDGE_SECONDS * FS

    rows = []
    for i, man_path in enumerate(manual_csvs):
        nk_path = nk2_dir / man_path.parent.name / man_path.name
        if not nk_path.exists():
            print(f"  manjka {nk_path}, preskakujem")
            continue

        man = pd.read_csv(man_path)
        nk = pd.read_csv(nk_path)
        if len(man) != len(nk) or len(man) < 3 * edge:
            print(f"  {man_path.name}: neujemajoca dolzina ali prekratek posnetek, preskakujem")
            continue

        row = {"file": man_path.name}
        for name, (bio_col, sen_col) in COMPONENTS.items():
            # primerjamo obe napravi posebej in porocamo slabsi (bolj konzervativni) rezultat
            spearmans, max_diffs = [], []
            for col in (bio_col, sen_col):
                a = interior(man[col], edge)
                b = interior(nk[col], edge)
                spearmans.append(pd.Series(a).corr(pd.Series(b), method="spearman"))
                max_diffs.append(np.abs(a - b).max())
            row[f"{name}_spearman"] = min(spearmans)
            row[f"{name}_max_abs_diff"] = max(max_diffs)
        rows.append(row)
        print(f"[{i + 1}/{len(manual_csvs)}] {man_path.name}")

    df = pd.DataFrame(rows)
    out = output_dir / "manual_vs_nk2.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 60)
    print(f"Primerjava rocno vs NeuroKit2, notranjost signala (brez {EDGE_SECONDS} s na robovih)")
    print(f"Stevilo meritev: {len(df)}")
    for name in COMPONENTS:
        sp = df[f"{name}_spearman"]
        md = df[f"{name}_max_abs_diff"]
        print(f"  {name:7s} Spearman: min={sp.min():.6f}  mediana={sp.median():.6f}")
        print(f"  {name:7s} max|razlika|: najvecja={md.max():.3e}")
    print("=" * 60)
    print(f"Shranjeno v {out}")


if __name__ == "__main__":
    main()
