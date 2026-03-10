from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import LABEL_COLUMNS, META_COLUMNS

def parse_csv(uploaded) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(uploaded)
    except Exception:
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, sep=";")
        except Exception:
            return None

def dataset_summary(df: Optional[pd.DataFrame]) -> Tuple[Optional[int], Optional[int]]:
    if df is None:
        return None, None

    cols = list(df.columns)
    label_cols = [c for c in cols if c.lower() in LABEL_COLUMNS]
    feature_cols = [c for c in cols if c not in label_cols]
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