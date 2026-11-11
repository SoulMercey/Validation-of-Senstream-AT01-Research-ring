import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_bland_altman_subplot(ax, b, s, title, unit):
    """Fills a matplotlib axis with a Bland-Altman plot, handling NaNs."""
    # Ensure we are working with clean, aligned data
    mask = ~np.isnan(b) & ~np.isnan(s)
    b = b[mask]
    s = s[mask]
    
    if len(b) == 0:
        ax.set_title(f"{title} (No Data)")
        return

    # Bland-Altman Math
    mean = (b + s) / 2
    diff = b - s
    md = np.mean(diff)
    sd = np.std(diff)
    
    # Scatter points
    ax.scatter(mean, diff, alpha=0.5, color='blue', s=1)
    
    # Statistical lines (dotted)
    ax.axhline(md, color='red', linestyle=':', label=f'Bias: {md:.4f}')
    ax.axhline(md + 1.96*sd, color='blue', linestyle=':', label='Limits of Agreement (95%)')
    ax.axhline(md - 1.96*sd, color='blue', linestyle=':')
    
    ax.set_title(title)
    ax.set_xlabel(f'Mean Signal ({unit})')
    ax.set_ylabel(f'Difference (B-S) ({unit})')
    ax.legend(loc='upper right', fontsize='small')

def single_subject_bland_altman(file_path):
    """Processes a single pre-filtered file and displays Phasic/Tonic Bland-Altman subplots."""
    try:
        # Load data and drop any rows that don't have our processed components
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required = ['Biopac_Phasic', 'Senstream_Phasic', 'Biopac_Tonic', 'Senstream_Tonic']
        if not all(col in df.columns for col in required):
            print(f"Skipping {file_path}: Missing required columns.")
            return False
            
        filename = os.path.basename(file_path).replace('.csv', '')
        
        # Create figure with 2 subplots (side-by-side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'Bland-Altman Analysis (Pre-filtered): {filename}', fontsize=16)

        # Plot 1: Phasic
        b_p = df['Biopac_Phasic'].values
        s_p = df['Senstream_Phasic'].values
        generate_bland_altman_subplot(ax1, b_p, s_p, 'Phasic Component Agreement', 'z-score')

        # Plot 2: Tonic
        b_t = df['Biopac_Tonic'].values
        s_t = df['Senstream_Tonic'].values
        generate_bland_altman_subplot(ax2, b_t, s_t, 'Tonic Component Agreement', 'z-score')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Interactive display
        plt.show()
        plt.close()

        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main(base_path):
    print(f"Starting Bland-Altman analysis for data in: {base_path}")
    if not os.path.exists(base_path):
        print(f"Error: Directory {base_path} does not exist.")
    else:
        # Find all CSV files in the filtered folder and its subfolders
        files = []
        for root, _, filenames in os.walk(base_path):
            for fn in filenames:
                if fn.lower().endswith('.csv'):
                    files.append(os.path.join(root, fn))

        print(f"Displaying {len(files)} individual Bland-Altman plots from {base_path} (recursively)...")
        print("Close the plot window to proceed to the next file.")
            
        for i, f_path in enumerate(sorted(files)):
            print(f"[{i+1}/{len(files)}] {os.path.relpath(f_path, base_path)}")
            ok = single_subject_bland_altman(f_path)
            if not ok:
                print(f"  Skipped: {os.path.basename(f_path)} (missing columns or read error)")
        
        print("\n[Done] All plots displayed.")


if __name__ == "__main__":
    base_path = 'Meritve/Filtered/manual'
    main(base_path)
