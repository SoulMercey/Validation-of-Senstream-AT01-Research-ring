# Masters thesis at ULFE: Validation of Senstream AT01 Research ring

This repository contains the measurement data and Python analysis workflow for evaluating the Senstream AT01 research ring against a Biopac MP150 laboratory reference. It provides the recordings, processing scripts, statistical comparisons, and plotting tools used to investigate how well the wearable captures electrodermal activity (EDA).


The repository can be used to inspect paired recordings, reproduce the signal comparisons, and generate figures. The [full thesis](thesis/Magistrska%20Vukadin%20final.pdf) describes the experimental protocol, processing methods, results, and limitations.

## Abstract

In this master's thesis, we investigated whether the Senstream AT01 sensor ring is a suitable alternative to the Biopac MP150 laboratory system with an EDA100C module for measuring electrodermal activity (EDA) when mobility and ease of use are important.

A total of 46 participants took part in the study. The ring was placed on a finger of the dominant hand, while the Biopac electrodes were attached to the index and middle fingers of the non-dominant hand. The measurement protocol comprised two recordings per participant: one while seated and one while standing. Each recording included two tasks involving hyperventilation: the first while remaining still and the second with additional arm movement. The analysis included 85 recordings, of which 41 were obtained while seated and 44 while standing.

The results showed that the EDA signal measured by the ring closely followed the signal recorded by the reference system. The mean cross-correlation coefficient between the raw signals was 0.86 and 0.87 for the first and second tasks, respectively. The time courses of the tonic and phasic components were also similar. Agreement in the detection of individual skin conductance responses was limited, as the two devices did not always detect the same responses. The sampling frequency of 5~Hz used for the ring may also have affected the detection of these responses.

**Based on the results, we consider the Senstream AT01 a suitable alternative** to a laboratory system for monitoring the time courses of the tonic and phasic components of the EDA signal when mobility and ease of use are important. Limitations remain in the detection of individual skin conductance responses. The built-in accelerometer proved useful for identifying periods of movement within the laboratory protocol used in this study.


## Run the analysis

Use Python 3.12. From the repository root, create and activate a virtual environment, install the dependencies, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python Scripts/run_all.py
```

The pipeline trims the raw recordings, validates recording pairs, generates combined CSVs, and runs the signal processing and statistical analyses. Outputs are written to `Meritve/Combined`, `Meritve/Filtered`, and `Results`.

## Analysis workflow

1. **Prepare paired recordings.** Trim BioPack and Senstream signals using their event markers, match recordings by day, participant ID, and measurement number, and create combined CSVs on a common 10 Hz time grid.
2. **Process the signals.** Apply manual filtering, NeuroKit2, and cvxEDA to obtain tonic and phasic components. The analyses also compare the wearable application's filtered signal with the reference.
3. **Evaluate agreement.** Calculate correlations, time-shifted correlations, signal errors, and peak-detection precision, recall, and F1 scores. Comparisons cover whole recordings and the protocol's rest, static-task, and movement-task segments.
4. **Summarize and visualize.** Export recording-level metrics, aggregate summaries, and paired statistical tests. Separate plotting scripts and functions generate signal comparisons, decomposition plots, and figures showing periods of movement.

## Directory structure

```text
Magistrska-anonymous/
├── Instructional video/               # Participant instructions used during the measurement protocol
│   ├── Instrucitional_Video_hyperventilate_PT1_FINAL.mp4
│   └── README.md                      # Note on the captions used in Part 2
├── Meritve/
│   ├── BioPack/                       # Raw reference recordings (.acq), grouped by day
│   ├── Senstream/                     # Raw wearable recordings and ZIP archives, grouped by day
│   ├── Combined/                      # Generated synchronized and resampled 10 Hz CSVs
│   └── Filtered/                      # Generated processed signals and decomposed components
│       ├── manual/                    # Manual filtering outputs (tonic and phasic)
│       ├── neurokit2/                 # NeuroKit2 processing outputs
│       └── cvxeda/                    # cvxEDA decomposition outputs
├── Results/                           # Generated metrics, summaries, and plots
│   ├── raw_correlation_all.csv         # Raw-signal correlations for each included recording
│   ├── peak_detection_all.csv          # Peak-detection metrics for each processing method
│   ├── *_quality_metrics*.csv          # Signal-quality metrics by method, posture, and segment
│   ├── Summary/
│   │   ├── raw_summary.csv            # Aggregate raw-signal metrics
│   │   ├── filtered_summary.csv       # Aggregate processed-signal metrics
│   │   ├── peak_summary.csv           # Aggregate peak-detection metrics
│   │   └── paired_tests.csv           # Paired statistical comparisons
│   ├── Graphs/                        # Figures produced by plotting functions
│   └── segment_plots/                 # Recording plots split into protocol segments
├── Scripts/                           # Preprocessing, analysis, and plotting scripts
├── thesis/
│   ├── Magistrska Vukadin final.pdf    # Full master's thesis
│   └── Slike/                         # Figures generated by thesis plotting functions
├── requirements.txt                   # Python dependencies
└── README.md                          # Project overview and run guide
```

Raw recordings in `BioPack` and `Senstream` are organized into `Dan 1`, `Dan 2`, and `Dan 3` (measurement days). Preprocessing creates `csv/` subfolders within those day folders. Raw Senstream signal CSVs are part of the input data; the day-level `csv/` subfolders contain trimmed outputs.

Generated data, results, and figures are created locally and ignored by Git. `python Scripts/run_all.py` produces the processed data and analysis tables; plotting scripts and functions produce the figure folders shown above.
