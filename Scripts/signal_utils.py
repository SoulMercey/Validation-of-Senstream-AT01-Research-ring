"""Skupni pripomocki za obdelavo signalov.

Namenjeno skriptam, ki delajo na _combined ali filtriranih datotekah in morajo
uporabiti enak casovni izsek kot analiza surovih signalov.
"""

import pandas as pd

BASELINE_SECONDS = 60


def trim_initial_seconds(df: pd.DataFrame, seconds: int = BASELINE_SECONDS) -> pd.DataFrame:
    """Odrezi zacetnih `seconds` sekund meritve (obdobje aklimatizacije).

    Za razliko od `calculate_correlation.filter_out_initial_60_seconds` ta razlicica
    ne predpostavlja stolpcev `biopac`/`senstream`, zato jo lahko uporabimo tudi na
    filtriranih datotekah s stolpci `Biopac_Tonic`, `Senstream_Phasic` ipd.

    Indeks je resetiran, ker detekcija vrhov in `Series.shift()` delujeta po poziciji.
    """
    if "time" not in df.columns:
        return df.reset_index(drop=True)

    df = df.copy()
    time = pd.to_datetime(df["time"], errors="coerce")
    valid = time.notna()
    if not valid.any():
        return df.reset_index(drop=True)

    elapsed = time - time[valid].iloc[0]
    keep = elapsed >= pd.to_timedelta(seconds, unit="s")
    return df.loc[keep].reset_index(drop=True)
