import pandas as pd
import os
import bioread
import matplotlib.pyplot as plt



#This script opens the senstream files. Checks where the first flag is and removes all data before it. The result is saved as a .csv file where we have EDA in respect to seconds

def main():
    parent_dir = ['Meritve/BioPack/Dan 1','Meritve/BioPack/Dan 2', 'Meritve/BioPack/Dan 3']

    for parent in parent_dir:
        files = os.listdir(parent)


        for file in files:
            path = f"{parent}/{file}"
            save_dir = f"{parent}/csv"
            #path = "Meritve/BioPack/Dan 2/P020 1.acq"
            print(path)
            
            #only read .acq files
            if not file.endswith('.acq'):
                continue
            
            #read file
            data = bioread.read(path)
            eda_channel = data.channels[0]
            eda_values = eda_channel.data
            time_axis = eda_channel.time_index

            #create df with which we will export trimmed data
            df = pd.DataFrame({'time': time_axis,'eda': eda_values})
            if len(data.event_markers) < 2:
                print(f"ERROR: File {path} path only has 1 flag \n {data.event_markers}\n")
                continue
            #first flag adter creation is our start flag
            start_time = data.event_markers[1].sample_index / eda_channel.samples_per_second
            #trim data
            df_trimmed = df[df["time"] >= start_time].copy()
            df_trimmed["time"] = df_trimmed["time"] - start_time
            #print(df_trimmed.head())
            
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(file)[0]
            df_trimmed.to_csv(f"{save_dir}/{base_name}.csv", index=False, header=False)

 
if __name__ == "__main__":
    main()

        
        
