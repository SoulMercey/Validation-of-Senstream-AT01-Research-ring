import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def normalize_time(df):
    """Converts time column to seconds starting from 0."""
    if 'time' in df.columns:
        try:
            df['time'] = pd.to_datetime(df['time'])
            df['time_sec'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
        except Exception as e:
            print(f"Error normalizing time: {e}")
            df['time_sec'] = range(len(df))
    return df

def visualize_files(file_list):
    bad_files = []
    current_idx = 0

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    def plot_current():
        nonlocal current_idx
        if current_idx >= len(file_list):
            print("\nAll files processed.")
            plt.close()
            return

        file_path = file_list[current_idx]
        print(f"Loading ({current_idx + 1}/{len(file_list)}): {file_path.name}")
        
        try:
            df = pd.read_csv(file_path)
            df = normalize_time(df)

            for ax in axs.flat:
                ax.clear()

            # Plot 1: All
            if 'biopac' in df.columns:
                axs[0, 0].plot(df['time_sec'], df['biopac'], label='Biopac', alpha=0.7)
            if 'senstream' in df.columns:
                axs[0, 0].plot(df['time_sec'], df['senstream'], label='Senstream', alpha=0.7)
            if 'senstream_filtered' in df.columns:
                axs[0, 0].plot(df['time_sec'], df['senstream_filtered'], label='Senstream Filtered', alpha=0.7)
            axs[0, 0].set_title('All Columns')
            axs[0, 0].legend()

            # Plot 2: Biopac
            if 'biopac' in df.columns:
                axs[0, 1].plot(df['time_sec'], df['biopac'], color='blue')
                axs[0, 1].set_title('Biopac Only')
            else:
                axs[0, 1].set_title('Biopac (Not found)')

            # Plot 3: Senstream
            if 'senstream' in df.columns:
                axs[1, 0].plot(df['time_sec'], df['senstream'], color='green')
                axs[1, 0].set_title('Senstream Only')
            else:
                axs[1, 0].set_title('Senstream (Not found)')

            # Plot 4: Senstream Filtered
            if 'senstream_filtered' in df.columns:
                axs[1, 1].plot(df['time_sec'], df['senstream_filtered'], color='red')
                axs[1, 1].set_title('Senstream Filtered Only')
            else:
                axs[1, 1].set_title('Senstream Filtered (Not found)')

            for ax in axs.flat:
                ax.set_xlabel('Čas (s)')
                ax.grid(True, linestyle='--', alpha=0.6)

            fig.suptitle(f"File ({current_idx + 1}/{len(file_list)}): {file_path.name}\n[Space] Next | [X] Mark Bad | [Esc] Exit")
            fig.canvas.draw()
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            current_idx += 1
            plot_current()

    def on_key(event):
        nonlocal current_idx
        if event.key == ' ':
            current_idx += 1
            plot_current()
        elif event.key == 'x' or event.key == 'X':
            bad_files.append(str(file_list[current_idx]))
            print(f"!!! Marked as bad: {file_list[current_idx].name}")
            current_idx += 1
            plot_current()
        elif event.key == 'escape':
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plot_current()
    plt.show()

    print("\n--- Summary of Bad Files ---")
    if bad_files:
        for f in bad_files:
            print(f)
    else:
        print("No files marked as bad.")

def main():
    # Discovery logic
    script_path = Path(__file__).resolve()
    # Assume script is in Magistrska/Scripts/
    base_dir = script_path.parent.parent / "Meritve" / "Combined"
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
    else:
        csv_files = sorted(list(base_dir.rglob("*.csv")))
        
        if not csv_files:
            print(f"No CSV files found in {base_dir}")
        else:
            visualize_files(csv_files)
if __name__ == "__main__":
    main()
