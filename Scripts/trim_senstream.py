import glob
import matplotlib.pyplot as plt
import pandas as pd
import os


def find_ring_dir(path):
    """Vrne mapo s signali prstana (ime vsebuje serijsko stevilko, npr. 'Senstream #27680').

    Prej je bila stevilka trdo zapisana, zato bi meritev z drugim prstanom tiho preskocilo.
    """
    matches = glob.glob(os.path.join(path, 'signalFiles', 'Senstream #*'))
    return matches[0] if matches else None


def trim(df, start_time):
    """Obrezi na zacetek testa in pretvori casovno os v sekunde od 0."""
    trimmed = df[df[0] >= start_time].copy()
    trimmed[0] = (trimmed[0] - start_time) * 10**-6
    return trimmed


#This script opens the acq file. Checks where the start flag is and removes all data before it. The result is saved as a .csv file where we have EDA in respect to seconds
def main():

    parent_dir = ['Meritve/Senstream/Dan 1','Meritve/Senstream/Dan 2','Meritve/Senstream/Dan 3']



    for parent in parent_dir:
        #print(parent)
        folders = os.listdir(parent)

        for folder in folders:
            #print(folder)
            #paths
            path = os.path.join(parent, folder)

            # preskoci vse, kar ni mapa meritve (Originals, lastna izhodna mapa csv, datoteke)
            if not os.path.isdir(path) or folder in ('Originals', 'csv'):
                continue

            ring_dir = find_ring_dir(path)
            if ring_dir is None:
                print(f"SKIP: missing signalFiles/Senstream #* folder for {folder}")
                continue

            file_eda_raw = os.path.join(ring_dir, 'edaRaw.csv')
            file_eda_filtered = os.path.join(ring_dir, 'edaFiltered.csv')
            file_accel = os.path.join(ring_dir, 'netAcceleration.csv')
            file_events = os.path.join(path, 'signalFiles', 'app', 'eventMarkers.csv')

            if not os.path.exists(file_eda_raw):
                print(f"SKIP: missing raw EDA file for {folder}: {file_eda_raw}")
                continue

            if not os.path.exists(file_eda_filtered):
                print(f"SKIP: missing filtered EDA file for {folder}: {file_eda_filtered}")
                continue

            if not os.path.exists(file_events):
                print(f"SKIP: missing eventMarkers for {folder}: {file_events}")
                continue

            #read and make into DF
            df_raw = pd.read_csv(file_eda_raw, header=None)
            df_prefiltered = pd.read_csv(file_eda_filtered, header=None)
            df_events = pd.read_csv(file_events, header=None)
            #get time of start of test
            start_time = df_events.iloc[0, 0]

            #trim data so that we only keep data from the actual test
            df_raw_trimmed = trim(df_raw, start_time)
            df_prefiltered_trimmed = trim(df_prefiltered, start_time)

            #save trimmed df as csvs
            save_path_raw = f"{parent}/csv/raw"
            save_path_filtered = f"{parent}/csv/filtered"

            os.makedirs(save_path_raw, exist_ok=True)
            os.makedirs(save_path_filtered, exist_ok=True)

            df_raw_trimmed.to_csv(f"{save_path_raw}/{folder}.csv", index=False, header=False)
            df_prefiltered_trimmed.to_csv(f"{save_path_filtered}/{folder}.csv", index=False, header=False)

            # Pospeskometer (50 Hz, norma pospeska v g z ze odstranjeno gravitacijo).
            # Deli isto uro kot EDA, zato velja isti start_time in je poravnava zagotovljena.
            if os.path.exists(file_accel):
                df_accel = pd.read_csv(file_accel, header=None)
                df_accel_trimmed = trim(df_accel, start_time)
                save_path_accel = f"{parent}/csv/accel"
                os.makedirs(save_path_accel, exist_ok=True)
                df_accel_trimmed.to_csv(f"{save_path_accel}/{folder}.csv", index=False, header=False)
            else:
                print(f"  opozorilo: ni netAcceleration.csv za {folder}")

            print(f"✓ Processed: {folder}")
        print(f"\n\n\n\n✓ Processed: {parent}")

if __name__ == "__main__":
    main()
