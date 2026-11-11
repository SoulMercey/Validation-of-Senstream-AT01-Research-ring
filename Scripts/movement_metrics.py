"""Jakost gibanja po segmentih iz pospeskometra v prstanu.

Stolpec `net_acceleration` v _combined datotekah je norma pospeska v g z ze odsteto
gravitacijo, prevzorcena na 10 Hz kot povprecje po 100 ms predalcku (glej make_combined.py).
Pri mirovanju je blizu nic, pri gibanju naraste.

Meje segmentov uvozimo iz correlation_by_parts, da so povsod enake.

Izhoda:
  Results/movement_metrics.csv          - vrstica na meritev x segment
  Results/Summary/movement_summary.csv  - povprecje +- SD in N po segmentu x pogoju
"""

from pathlib import Path

import pandas as pd

from correlation_by_parts import (
    BREAK_TIME_1_END,
    BREAK_TIME_2_START,
    BREAK_TIME_2_END,
)
from file_exclusions import filter_csv_paths

SEGMENTS = ("static", "break", "dynamic")


def segment_masks(elapsed):
    """Iste meje kot correlation_by_parts: static (60,110], break (110,160), dynamic >160."""
    return {
        "static": (elapsed > BREAK_TIME_1_END) & (elapsed <= BREAK_TIME_2_START),
        "break": (elapsed > BREAK_TIME_2_START) & (elapsed < BREAK_TIME_2_END),
        "dynamic": elapsed > BREAK_TIME_2_END,
    }


def condition(path):
    """Sede = meritev 1, stoje = meritev 2 (enako kot drugod v analizi)."""
    stem = Path(path).stem
    if stem.endswith("1_combined"):
        return "sitting"
    if stem.endswith("2_combined"):
        return "standing"
    return "other"


def segment_rows(path):
    """Povprecna jakost gibanja po segmentih za eno meritev; [] ce pospeska ni."""
    df = pd.read_csv(path)
    if "net_acceleration" not in df.columns:
        return []
    accel = pd.to_numeric(df["net_acceleration"], errors="coerce")
    if accel.isna().all():
        return []

    time = pd.to_datetime(df["time"], errors="coerce")
    masks = segment_masks(time - time.iloc[0])

    rows = []
    for segment in SEGMENTS:
        values = accel[masks[segment]].dropna()
        if values.empty:
            continue
        rows.append({
            "file": Path(path).name,
            "day": Path(path).parent.name,
            "condition": condition(path),
            "segment": segment,
            "mean_accel": values.mean(),
            "sd_accel": values.std(ddof=1),
            "n_samples": len(values),
        })
    return rows


def summarise(df):
    """Povprecje +- SD jakosti gibanja po pogoju x segmentu, s skupno vrstico 'all'."""
    both = pd.concat([df, df.assign(condition="all")], ignore_index=True)
    return both.groupby(["condition", "segment"], as_index=False).agg(
        n_files=("file", "count"),
        mean_accel_mean=("mean_accel", "mean"),
        mean_accel_std=("mean_accel", lambda s: s.std(ddof=1)),
    )


def report(df):
    """Izpise preizkus metrike in primerjavo odmora s 1. nalogo."""
    wide = df.pivot_table(index=["file", "condition"], columns="segment",
                          values="mean_accel").reset_index()

    # Preizkus: po protokolu je roka v 2. nalogi v gibanju, v 1. pa miruje. Ce to ne drzi,
    # je nekaj narobe s poravnavo pospeska in ne z meritvami.
    higher = int((wide["dynamic"] > wide["static"]).sum())
    ratio = wide["dynamic"].mean() / wide["static"].mean()
    print(f"\nPreizkus metrike: 2. naloga > 1. naloga pri {higher}/{len(wide)} meritvah, "
          f"razmerje povprecij {ratio:.1f}x")
    if higher < 0.7 * len(wide) or ratio < 1.5:
        print("  POZOR: preizkus ni prepricljiv - preveri poravnavo pospeska!")

    print("\nPovprecna jakost gibanja [g] po segmentih:")
    print(df.pivot_table(index="condition", columns="segment",
                         values="mean_accel")[list(SEGMENTS)].round(4).to_string())

    # Domneva, da so se med odmorom brez naloge vec premikali. Izmerjeno je obratno.
    print("\nOdmor proti 1. nalogi:")
    for cond in ("sitting", "standing"):
        sub = wide[wide["condition"] == cond]
        n_more = int((sub["break"] > sub["static"]).sum())
        print(f"  {cond:9s}: odmor {sub['break'].mean():.4f} g proti "
              f"1. nalogi {sub['static'].mean():.4f} g; "
              f"odmor vecji pri {n_more}/{len(sub)} meritvah")


def main():
    base = Path(__file__).resolve().parent.parent
    results_dir = base / "Results"
    summary_dir = results_dir / "Summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    files = filter_csv_paths(sorted((base / "Meritve" / "Combined").rglob("*_combined.csv")))
    print(f"Datotek po izkljucitvah: {len(files)}")

    rows = []
    for path in files:
        file_rows = segment_rows(path)
        if not file_rows:
            print(f"  BREZ POSPESKA: {path.name}")
        rows.extend(file_rows)

    if not rows:
        print("Ni meritev s pospeskom, koncujem.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "movement_metrics.csv", index=False)
    print(f"Shranjeno: {results_dir / 'movement_metrics.csv'}  ({len(df)} vrstic)")

    summarise(df).to_csv(summary_dir / "movement_summary.csv", index=False)
    print(f"Shranjeno: {summary_dir / 'movement_summary.csv'}")

    report(df)


if __name__ == "__main__":
    main()
