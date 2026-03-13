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


def parse_csv(uploaded) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(uploaded)
    except Exception:
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, sep=";")
        except Exception:
            return None


def parse_iowa_mat(uploaded, preprocessing_summary=None):
    """
    Transformă un fișier Iowa .mat într-un DataFrame tabular pentru SVM,
    folosind setările salvate în pagina Preprocess.
    """
    cfg = preprocessing_summary or {}

    fs = 1000.0
    window_sec = float(cfg.get("window_sec", RAW_WINDOW_SEC))
    step_sec = float(cfg.get("step_sec", RAW_STEP_SEC))
    max_windows = int(cfg.get("max_windows_per_recording", RAW_MAX_WINDOWS_PER_RECORDING))
    use_notch = bool(cfg.get("use_notch", RAW_USE_NOTCH))
    notch_freq = float(cfg.get("notch_freq", RAW_NOTCH_FREQ))
    use_bandpass = bool(cfg.get("use_bandpass", RAW_USE_BANDPASS))
    bandpass_low = float(cfg.get("bandpass_low", RAW_L_FREQ))
    bandpass_high = float(cfg.get("bandpass_high", RAW_H_FREQ))

    window = int(window_sec * fs)
    step = int(step_sec * fs)

    if window <= 0 or step <= 0:
        return None, None, "Invalid preprocessing configuration."

    tmp_path = None
    try:
        suffix = os.path.splitext(uploaded.name)[1] if getattr(uploaded, "name", None) else ".mat"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

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

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def dataset_summary(df: Optional[pd.DataFrame]) -> Tuple[Optional[int], Optional[int]]:
    if df is None:
        return None, None

    cols = list(df.columns)
    label_cols = [c for c in cols if c.lower() in LABEL_COLUMNS]
    feature_cols = [c for c in cols if c not in label_cols and c not in META_COLUMNS]
    return len(df), len(feature_cols)


def get_xy(df: pd.DataFrame):
    cols = list(df.columns)
    label_candidates = [c for c in cols if c.lower() in LABEL_COLUMNS]

    if not label_candidates:
        return None, None, "Dataset CSV must contain a label column: label/class/y/target."

    ycol = label_candidates[0]
    X = df.drop(columns=[ycol]).copy()
    X = X.drop(columns=[c for c in META_COLUMNS if c in X.columns], errors="ignore")
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