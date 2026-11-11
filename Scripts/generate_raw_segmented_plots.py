from pathlib import Path
import os
import pandas as pd
import numpy as np
import importlib.util
import matplotlib

if os.environ.get("DISPLAY") is None and os.name == "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location('corr', 'Scripts/correlation_by_parts.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

base = Path('Meritve/Combined')
files = sorted(base.rglob('*2_combined.csv'))
print(len(files), 'files found')
rows = []
output_dir = Path('Results/segment_plots')
output_dir.mkdir(parents=True, exist_ok=True)

for p in files:
    df = pd.read_csv(p)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['biopac'] = pd.to_numeric(df['biopac'], errors='coerce')
    df['senstream'] = pd.to_numeric(df['senstream'], errors='coerce')
    df = df.dropna(subset=['time', 'biopac', 'senstream'])
    elapsed = df['time'] - df['time'].iloc[0]

    masks = {
        'full': np.ones(len(df), dtype=bool),
        'static': (elapsed > mod.BREAK_TIME_1_END) & (elapsed <= mod.BREAK_TIME_2_START),
        'dynamic': elapsed > mod.BREAK_TIME_2_END,
    }

    fig, axes = plt.subplots(3, 2, figsize=(18, 10), sharex=False)
    fig.suptitle(f"{p.name}", fontsize=14, fontweight='bold')

    for row_idx, (name, mask) in enumerate(masks.items()):
        if row_idx == 0:
            segment_df = df
        else:
            segment_df = df[mask]

        if segment_df.empty:
            for col_idx in range(2):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, f'No data for {name}', ha='center', va='center')
                ax.set_title(name.capitalize())
                ax.set_ylabel('Value')
            continue

        for col_idx, signal_name in enumerate(['senstream', 'biopac']):
            ax = axes[row_idx, col_idx]
            ax.plot(segment_df['time'], segment_df[signal_name], color='tab:blue' if signal_name == 'biopac' else 'tab:orange', linewidth=1.2)
            ax.set_title(f"{name.capitalize()} - {signal_name.capitalize()}")
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)

            if name != 'full':
                corr = segment_df['biopac'].corr(segment_df['senstream'], method='spearman')
                ax.text(0.98, 0.02, f'Spearman: {corr:.3f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=9)

    for row_idx in range(3):
        axes[row_idx, 0].set_ylabel('Senstream')
        axes[row_idx, 1].set_ylabel('Biopac')
        axes[row_idx, 0].set_xlabel('Time')
        axes[row_idx, 1].set_xlabel('Time')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = output_dir / f"{p.stem}_segments.png"
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot for {p.name} -> {save_path}')

    for seg, mask in masks.items():
        if seg == 'full':
            continue
        s = df[mask]
        if not s.empty:
            rows.append((p.name, seg, s['biopac'].corr(s['senstream'], method='spearman')))

print('rows', len(rows))
for row in rows[:20]:
    print(row)

by_seg = {}
for _, seg, corr in rows:
    by_seg.setdefault(seg, []).append(corr)
for seg, vals in by_seg.items():
    print(seg, 'count', len(vals), 'mean', np.nanmean(vals))
