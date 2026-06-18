from typing import Optional, Tuple
import os
import tempfile

import pandas as pd

from src.config import (
    LABEL_COLUMNS,
    META_COLUMNS,
    RAW_WINDOW_SEC,
    RAW_STEP_SEC,
    RAW_MAX_WINDOWS_PER_RECORDING,
    RAW_L_FREQ,
    RAW_H_FREQ,
    RAW_NOTCH_FREQ,
    RAW_USE_BANDPASS,
    RAW_USE_NOTCH,
)
from scripts.make_features_iowa import build_iowa_features_from_mat


def parse_csv(uploaded) -> Optional[pd.DataFrame]: # reads csv file
    try:
        return pd.read_csv(uploaded)
    except Exception:
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, sep=";")
        except Exception:
            return None


def parse_iowa_mat(uploaded, preprocessing_summary=None): # converts .mat into .csv
    # uses preprocessing settings from the app
    cfg = preprocessing_summary or {}

    # loads sig segmentation settings
    fs = 1000.0 # sampling rate
    window_sec = float(cfg.get("window_sec", RAW_WINDOW_SEC)) # window length
    step_sec = float(cfg.get("step_sec", RAW_STEP_SEC)) # step size
    max_windows = int(cfg.get("max_windows_per_recording", RAW_MAX_WINDOWS_PER_RECORDING))
    # loads notch filtering settings
    use_notch = bool(cfg.get("use_notch", RAW_USE_NOTCH))
    notch_freq = float(cfg.get("notch_freq", RAW_NOTCH_FREQ))
    # loads bandpass filtering settings
    use_bandpass = bool(cfg.get("use_bandpass", RAW_USE_BANDPASS))
    bandpass_low = float(cfg.get("bandpass_low", RAW_L_FREQ))
    bandpass_high = float(cfg.get("bandpass_high", RAW_H_FREQ))

    # converts seconds to samples
    window = int(window_sec * fs)
    step = int(step_sec * fs)

    if window <= 0 or step <= 0:
        return None, None, "Invalid preprocessing configuration."

    tmp_path = None
    try:
        suffix = os.path.splitext(uploaded.name)[1] if getattr(uploaded, "name", None) else ".mat"

        # creates temp .mat file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            if hasattr(uploaded, "getbuffer"):
                tmp.write(uploaded.getbuffer())
            elif hasattr(uploaded, "getvalue"):
                tmp.write(uploaded.getvalue())
            else:
                tmp.write(uploaded.read())
            tmp_path = tmp.name
        # extracts feats from file
        df, summary = build_iowa_features_from_mat(
            mat_path=tmp_path,
            fs=fs,
            window=window,
            step=step,
            max_windows_per_subject=max_windows,
            use_bandpass=use_bandpass,
            use_notch=use_notch,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            notch_freq=notch_freq,
            verbose=False,
        )

        return df, summary, None

    except Exception as e:
        return None, None, str(e)

    # deletes temp file
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# returns basic info about the dataset
def dataset_summary(df: Optional[pd.DataFrame]) -> Tuple[Optional[int], Optional[int]]:
    if df is None:
        return None, None

    cols = list(df.columns)
    label_cols = [c for c in cols if c.lower() in LABEL_COLUMNS]
    candidate_cols = [c for c in cols if c not in label_cols and c not in META_COLUMNS]

    numeric_df = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    feature_cols = [c for c in numeric_df.columns if not numeric_df[c].isna().all()]

    return len(df), len(feature_cols)


def get_xy(df: pd.DataFrame):
    cols = list(df.columns)
    label_candidates = [c for c in cols if c.lower() in LABEL_COLUMNS]

    if not label_candidates:
        return None, None, "Dataset CSV must contain a label column: label/class/y/target."

    ycol = label_candidates[0]

    X = df.drop(columns=[ycol]).copy()
    X = X.drop(columns=[c for c in META_COLUMNS if c in X.columns], errors="ignore")

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    X = X.replace([float("inf"), float("-inf")], pd.NA)

    if X.shape[1] == 0:
        return None, None, "No numeric feature columns were found. Make sure you generated SVM features, not only the raw EEG manifest."

    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    y = df[ycol].copy()

    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower()
        y = y.map({"pd": 1, "hc": 0, "1": 1, "0": 0}).fillna(y)

    try:
        y = y.astype(int)
    except Exception:
        uniq = sorted(pd.unique(y))
        mapping = {v: i for i, v in enumerate(uniq)}
        y = y.map(mapping).astype(int)

    return X, y, None