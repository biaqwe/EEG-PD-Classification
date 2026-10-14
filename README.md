# EEG-Based Parkinson's Disease Classification

A Python-based machine learning application for **classifying Parkinson's disease (PD) and healthy control (HC) subjects from electroencephalography (EEG) data**. It combines traditional machine learning on extracted EEG features with deep learning on EEG spectrograms, and provides a **Streamlit dashboard** for importing data, configuring preprocessing, training models, and exploring results.

The project emphasizes **subject-aware evaluation**: windows from the same participant are kept together during training and testing to reduce data leakage and provide a more meaningful estimate of model performance.

## Features

- **Interactive Streamlit interface** with Dashboard, Import, Preprocess, Raw EEG Viewer, and Results pages.
- **Multiple EEG input formats:** feature tables (`.csv`), supported MATLAB EEG datasets (`.mat`), BrainVision recordings (`.vhdr`, `.eeg`, `.vmrk`), and EEGLAB recordings (`.set`, `.fdt`).
- **Configurable signal preprocessing:** window length and overlap, band-pass and notch filters, and spectrogram generation settings.
- **Two classification approaches:** a Support Vector Machine (SVM) on numerical EEG features and a Convolutional Neural Network (CNN) on EEG spectrograms.
- **Subject-aware train/test splitting and cross-validation**, including window-level and subject-level evaluation.
- **Model diagnostics and visualizations:** accuracy, F1-score, ROC AUC (when defined), confusion matrices, fold summaries, and individual prediction details.
- **Experiment tracking:** local JSON run records, execution logs, and a downloadable report for the latest run.

## Technologies

**Python**, **Streamlit**, **NumPy**, **pandas**, **SciPy**, **scikit-learn**, **PyTorch**, **MNE-Python**, **h5py**, and **Matplotlib**.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/biaqwe/EEG-PD-Classification.git
cd EEG-PD-Classification
```

### 2. Create a virtual environment

Python **3.11 or 3.12** is recommended for the pinned dependencies.

```bash
python -m venv .venv
```

Activate it on **Windows**:

```powershell
.venv\Scripts\activate
```

Or on **macOS / Linux**:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Installing PyTorch and scientific computing dependencies may require additional disk space. A GPU can accelerate CNN training, but the application also supports CPU execution.

### 4. Start the application

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal (normally `http://localhost:8501`).

## Using the Application

### Option A: SVM with the included feature dataset

The repository includes `dataset_pd_hc.csv`, which can be used to try the tabular classification pipeline.

1. Open **Import** and select **CSV feature table** under *SVM source*.
2. Upload `dataset_pd_hc.csv` and click **Load SVM dataset**.
3. Optionally review the imported features and labels.
4. Click **Train SVM** for a subject-aware train/test evaluation, or **Run SVM Group CV** for grouped cross-validation.
5. Open **Results** to inspect metrics, confusion matrices, diagnostics, and available prediction details.

The SVM pipeline uses **standardization**, **univariate feature selection** (`SelectKBest`), and an **RBF-kernel SVM** with balanced class weights.

### Option B: CNN with raw EEG recordings

1. Open **Import** and upload the files for your recordings:
   - **BrainVision:** matching `.vhdr`, `.eeg`, and `.vmrk` files.
   - **EEGLAB:** matching `.set` and `.fdt` files.
   - Optionally include a participant metadata file (`participants.tsv` or a supported CSV) to help identify subjects and labels.
2. Click **Load raw EEG dataset** and check the recording manifest for complete files, participant IDs, and inferred labels.
3. Open **Preprocess** to configure EEG filtering, windowing, and spectrogram parameters, then save the configuration.
4. Optionally use **Raw EEG Viewer** to inspect preprocessed signals.
5. Click **Train CNN** or **Run CNN Group CV**.
6. Examine the evaluation output in **Results**.

The CNN converts EEG windows into **time-frequency spectrograms** and learns patterns using a compact convolutional network. Spectrogram channels can be configured as `mean`, `regions`, or `all`.

**Important:** Check that subjects and PD/HC labels were identified correctly before training. The importer can infer labels from filenames or participant metadata; incorrect labels will invalidate the evaluation.

### Option C: MATLAB EEG data for SVM

On the **Import** page, select **EEG .mat file**, upload a supported EEG MATLAB dataset, and click **Load SVM dataset**. The application converts the signals into a tabular feature dataset using the configured preprocessing settings.

The MATLAB importer expects the EEG data structure handled by `scripts/make_features_mat.py`; it is **not a generic reader for every `.mat` layout**. Its current conversion path assumes a sampling frequency of **1,000 Hz**.

## Evaluation

The application provides two ways to evaluate each model:

| Mode | Description |
| --- | --- |
| **Train SVM** | Subject-grouped train/test split for a single SVM evaluation. |
| **Run SVM Group CV** | Stratified, subject-grouped cross-validation for the SVM. |
| **Train CNN** | Subject-separated training, validation, and testing of a spectrogram CNN. |
| **Run CNN Group CV** | Stratified, subject-grouped cross-validation with an inner validation split for the CNN. |

Cross-validation uses **up to five folds**, depending on the number of available subjects in each class. The Results page reports **window-level** predictions (for individual EEG segments) and **subject-level** predictions (aggregated across a participant's windows) where available.

Metrics include accuracy, F1-score, and ROC AUC where defined; group-CV diagnostics also include additional measures such as sensitivity, specificity, and balanced accuracy. Confusion matrices and fold details help inspect the outcomes beyond a single score.

Keeping each subject's windows within one partition helps prevent the model from being evaluated on recordings from a participant it has already seen during training.

## Dataset Format

For the SVM, a CSV must contain numerical feature columns, a **class label**, and a **`subject_key`** column for subject-grouped evaluation. Supported class-label column names are `label`, `class`, `y`, and `target`.

Typical metadata columns include `subject_id`, `subject_key`, `group`, and `window_start`. These are not used as numerical model features. By convention:

- `0` = healthy control (**HC**)
- `1` = Parkinson's disease (**PD**)

You can use the included `dataset_pd_hc.csv` as an example of the expected feature-table structure. For raw recordings, provide complete recording files and confirm that the application associates every recording with the correct subject and class.

## Optional Command-Line Experiments

The repository also contains scripts for running SVM experiments without the Streamlit interface:

```bash
python experiments/train_svm_baseline_subject_split.py
python experiments/train_svm_group_cv.py
```

Both scripts read `dataset_pd_hc.csv` from the repository root. They generate `svm_test_predictions.csv` and `svm_group_cv_results.json`, respectively. These are **separate experimental implementations**; for the current integrated application workflow, use the training controls in Streamlit.

To create an SVM feature table from a compatible MATLAB EEG file through the command line:

```bash
python -m scripts.make_features_mat --input path/to/dataset.mat --output extracted_features.csv
```

Additional preprocessing options are available through the script's command-line arguments.

## Project Structure

```text
EEG-PD-Classification/
├── app.py                         # Streamlit entry point
├── dataset_pd_hc.csv              # Included EEG feature table
├── requirements.txt               # Python dependencies
├── experiments/
│   ├── train_svm_baseline_subject_split.py
│   └── train_svm_group_cv.py
├── scripts/
│   └── make_features_mat.py       # MATLAB EEG feature extraction
└── src/
    ├── actions.py                # UI training actions
    ├── cnn_group_cv.py           # CNN grouped cross-validation
    ├── config.py                 # Paths and model defaults
    ├── data_utils.py             # CSV and MAT dataset handling
    ├── ml_models.py              # SVM pipeline and model utilities
    ├── raw_eeg.py                # Raw recording loading and EEG processing
    ├── simple_cnn.py             # Spectrogram CNN training
    ├── state.py                  # Streamlit session state
    ├── storage.py                # Local experiment records
    └── ui/                       # Pages, navigation, and styling
```

The application automatically creates local `runs/` and `results/` directories. Run metadata is saved in `runs/` as JSON, and the latest run record can be downloaded through the **Results → Export** tab.

## Limitations and Research Notes

- Results depend on the dataset, participant distribution, recording quality, preprocessing, and validation strategy. **No performance figure should be interpreted as clinical diagnostic accuracy without independent validation.**
- The included feature dataset provides an example input, but its provenance and data-collection protocol are not documented in this repository.
- Small or imbalanced cohorts can make subject-level metrics unstable, even with grouped cross-validation.
- MATLAB import is tailored to a specific EEG data structure, and raw EEG imports require compatible recording files and reliable participant labels.
- CNN training can be computationally intensive, especially with multiple cross-validation folds.

**Disclaimer:** This project is intended for educational and research purposes only. It is not a medical device and must not be used to diagnose Parkinson's disease or guide clinical decisions.
