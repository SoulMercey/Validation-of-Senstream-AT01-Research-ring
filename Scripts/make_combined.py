from pathlib import Path
import pandas as pd
import matplotlib.pyplot as pltf
import os
from pprint import pprint
from check_pairity import compare_files

def join_dfs(df_biopac, df_senstream, df_senstream_filtered, df_accel=None):
    #make sure time starts at 0
    df_biopac[0] = df_biopac.iloc[:,0] - df_biopac.iloc[0,0]
    df_senstream[0] = df_senstream.iloc[:,0] - df_senstream.iloc[0,0]
    df_senstream_filtered[0] = df_senstream_filtered.iloc[:,0] - df_senstream_filtered.iloc[0,0]

    #make columns pd datetimes for resampling
    df_biopac[0] = pd.to_datetime(df_biopac[0], unit='s')
    df_senstream[0] = pd.to_datetime(df_senstream[0], unit='s')
    df_senstream_filtered[0] = pd.to_datetime(df_senstream_filtered[0], unit='s')

    

    #resample so that dataframes match we need > 6hz at least for use with neurokit2 
    df_biopac_resampled = df_biopac.resample('100ms', on=0).mean().reset_index()
    df_senstream_resampled = df_senstream.resample('100ms', on=0).mean().interpolate().reset_index()
    df_senstream_filtered_resampled = df_senstream_filtered.resample('100ms', on=0).mean().interpolate().reset_index()
    #print(df_senstream_filtered_resampled.head())
    #join dataframes on time
    df_joined = df_biopac_resampled.join(
        df_senstream_resampled.iloc[:,1], 
        how='outer', 
        lsuffix='_biopac', 
        rsuffix='_senstream'
    )
    df_join_all = df_joined.join(
        df_senstream_filtered_resampled.iloc[:,1],
        how='outer',
        lsuffix='_biopac', 
        rsuffix='_senstream_filtered'
    )
    columns = ['time', 'biopac', 'senstream', 'senstream_filtered']
    df_join_all.columns = columns
    # dropna izvedemo SAMO nad temi stirimi stolpci; ce bi pospesek vkljucili in bi pri
    # kaksni meritvi manjkal, bi bil stolpec sam NaN in izbrisalo bi vse vrstice
    df_join_all.dropna(inplace=True)

    # Pospeskometer: 50 Hz, norma pospeska v g z ze odstranjeno gravitacijo.
    # Povprecje po 100 ms predalcku je pravilna agregacija - povprecje nenegativne norme
    # ohrani jakost gibanja, medtem ko bi interpolacija dala trenutno vrednost.
    if df_accel is not None:
        df_accel = df_accel.copy()
        df_accel[0] = df_accel.iloc[:, 0] - df_accel.iloc[0, 0]
        df_accel[0] = pd.to_datetime(df_accel[0], unit='s')
        df_accel_resampled = df_accel.resample('100ms', on=0).mean().interpolate().reset_index()
        accel = df_accel_resampled.iloc[:, 1]
        accel.name = 'net_acceleration'
        df_join_all = df_join_all.join(accel, how='left')
    else:
        df_join_all['net_acceleration'] = float('nan')

    return df_join_all

def main():
    paired, unpaired, biopacl_en, senstream_len, keys, duplicates = compare_files()
    if unpaired or duplicates:
        print(f"Combining only {len(paired)} unique, complete pairs; {len(unpaired)} unmatched and {len(duplicates)} duplicate keys excluded.")

    combined_root = Path(__file__).resolve().parent.parent / "Meritve" / "Combined"
    expected = {
        (combined_root / pair['day'] / f"{pair['name']}_combined.csv").resolve()
        for pair in paired
    }
    removed = 0
    if combined_root.exists():
        for old_csv in combined_root.rglob("*_combined.csv"):
            if old_csv.resolve() not in expected:
                old_csv.unlink()
                removed += 1
    if removed:
        print(f"Removed {removed} stale combined CSV(s) for unmatched or duplicate recordings.")

    for pair in paired:
        print(f"Processing pair: {pair['name']} from day {pair['day']}")
        df_biopac = pd.read_csv(pair['biopac_path'], header=None)
        df_senstream = pd.read_csv(pair['senstream_paths']['raw'], header=None)
        df_senstream_filtered = pd.read_csv(pair['senstream_paths']['filtered'], header=None)
        accel_path = pair['senstream_paths'].get('accel')
        df_accel = pd.read_csv(accel_path, header=None) if accel_path else None
        if df_accel is None:
            print(f"  opozorilo: ni pospeska za {pair['name']}")
        df_joined = join_dfs(df_biopac, df_senstream, df_senstream_filtered, df_accel)
        #save joined to combined folder in the same day folder
        day_folder = pair['day']
        save_path = combined_root / day_folder / f"{pair['name']}_combined.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_joined.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()
        
        

