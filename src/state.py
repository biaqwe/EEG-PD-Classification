from dataclasses import dataclass
from typing import Optional

import streamlit as st


@dataclass
class PreprocConfig: # stores preprocessing settings
    bandpass_low: float = 0.5 # lower bandpass freq
    bandpass_high: float = 40.0 # upper bandpass freq
    notch: float = 50.0 # notch filter freq
    epoch_sec: float = 2.0 # window length in sec
    normalize: str = "z-score" # normalization method


@dataclass
class RunRecord: # stores info about one run
    run_id: str
    timestamp: str
    dataset_name: Optional[str] # name of dataset used
    n_rows: Optional[int] # nr of rows in dataset
    n_channels: Optional[int] # nr of eeg channels
    preproc: dict # preprocessing config
    action: str # what was run
    status: str # result status
    metrics: dict


def init_session_state(): # default session state values
    defaults = {
        "page": "Dashboard", # default page
        "dataset_df": None, # loaded .csv dataset
        "dataset_name": None, # name of currently loaded dataset
        "preproc": PreprocConfig(), # default preprocessing config
        "run_status": "Idle", # app status
        "logs": [], # list of execution logs
        "last_metrics": {}, # metrics from last run
        "last_cm": None, # last confusion matrix
        "last_cm_window": None, # last window level confusion matrix
        "last_cm_subject": None, # last subject level confusion matrix
        "last_roc": None, # last roc curve
        "last_action": None, # last action run
        "preprocessing_summary": None, # saved preprocessing settings
        "preprocessing_logs": [], # saved preprocessing config logs
        "last_model": None, # last trained model
        "last_group_cv_predictions": None, # last svm group cv predictions

        "raw_file_payloads": None, # uploaded raw eeg files
        "raw_manifest_df": None, # table describil detected eeg recordings
        "raw_dataset_summary": None, # summary of raw eeg dataset
        "raw_cnn_predictions": None, # prediction table from cnn
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log(msg: str, now_iso_fn): # adds messages to logs
    st.session_state.logs.append(f"[{now_iso_fn()}] {msg}")


def set_status(new_status: str): # sets app status
    st.session_state.run_status = new_status