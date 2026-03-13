from dataclasses import dataclass
from typing import Optional

import streamlit as st


@dataclass
class PreprocConfig:
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0
    notch: float = 50.0
    epoch_sec: float = 2.0
    normalize: str = "z-score"


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    dataset_name: Optional[str]
    n_rows: Optional[int]
    n_channels: Optional[int]
    preproc: dict
    action: str
    status: str
    metrics: dict


def init_session_state():
    defaults = {
        "page": "Dashboard",
        "dataset_df": None,
        "dataset_name": None,
        "preproc": PreprocConfig(),
        "run_status": "Idle",
        "logs": [],
        "last_metrics": {},
        "last_cm": None,
        "last_cm_window": None,
        "last_cm_subject": None,
        "last_roc": None,
        "last_action": None,
        "preprocessing_summary": None,
        "preprocessing_logs": [],
        "last_model": None,
        "last_group_cv_predictions": None,

        "raw_file_payloads": None,
        "raw_manifest_df": None,
        "raw_dataset_summary": None,
        "raw_cnn_predictions": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log(msg: str, now_iso_fn):
    st.session_state.logs.append(f"[{now_iso_fn()}] {msg}")


def set_status(new_status: str):
    st.session_state.run_status = new_status