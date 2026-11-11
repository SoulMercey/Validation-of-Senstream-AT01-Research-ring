# EDA wearable validation

Raw BioPack and Senstream recordings, the original analysis scripts, and the final thesis PDF.

## Run the analysis

Use Python 3.12. From the repository root, create and activate a virtual environment, install the dependencies, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python Scripts/run_all.py
```

The existing entry point trims the raw recordings, reports pairing coverage, builds combined CSVs, and runs the analysis. Generated trimmed CSVs, combined CSVs, filtered data, results, and graphs are ignored by Git and are regenerated locally. CSV files inside the raw Senstream recordings remain included as source data.

## Anonymous recording names

Each participant within a measurement day has a unique ID (`P001` through `P046`). Corresponding BioPack and Senstream recordings use the same ID and measurement suffix (`1` or `2`) under the same day. The two different people who shared an original name on different days have separate IDs. Original name-to-ID mappings are not included.

Raw recordings and archives are retained, including recordings the existing pairing code excludes. Device identifiers, timestamps, signal samples, and event markers are preserved. Analysis and plotting logic is unchanged; participant-specific recording references were replaced with the anonymous names.

## Graphs and thesis PDF

Figures are generated locally under `thesis/Slike` and `Results` and are not committed. Run the analysis first. The existing plotting functions in `Scripts/analyze_results.py` retain their original calculations and styling. Its default entry point generates the selected Bland–Altman figures. Other existing plot entry points include:

```powershell
python Scripts/analyze_results.py
python Scripts/make_histograms.py
python Scripts/generate_raw_segmented_plots.py
```

`thesis/Magistrska Vukadin final.pdf` is copied unchanged from the original repository. The LaTeX project, presentation, separate consent/ethics documents, and instructional video are not included. The unchanged final PDF retains its original appendices.
