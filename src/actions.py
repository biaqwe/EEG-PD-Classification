import time

import streamlit as st

from src.ml_models import train_svm, train_svm_group_cv, fake_cnn_result
from src.state import set_status, log
from src.storage import save_run
from src.utils import now_iso


def require_dataset(action_name: str):
    if st.session_state.dataset_df is None:
        set_status("Error")
        log(f"Cannot run {action_name}: dataset not loaded.", now_iso)
        save_run(action=action_name, status="Error", metrics={"error": "dataset not loaded"})
        return False
    return True


def run_train_svm():
    if not require_dataset("svm"):
        return

    set_status("Running")
    st.session_state.last_action = "svm"
    log("Training SVM started.", now_iso)

    metrics, cm, roc, model, err = train_svm(st.session_state.dataset_df)

    if err:
        set_status("Error")
        log(f"SVM failed: {err}", now_iso)
        st.session_state.last_metrics = {"error": err}
        st.session_state.last_cm = None
        st.session_state.last_cm_window = None
        st.session_state.last_cm_subject = None
        st.session_state.last_roc = None
        st.session_state.last_model = None
        st.session_state.last_group_cv_predictions = None
        save_run(action="svm", status="Error", metrics={"error": err})
        return

    st.session_state.last_model = model
    st.session_state.last_group_cv_predictions = None

    st.session_state.last_metrics = metrics
    st.session_state.last_cm = cm
    st.session_state.last_cm_window = None
    st.session_state.last_cm_subject = None
    st.session_state.last_roc = roc

    set_status("Ready")
    log(f"SVM done. Metrics: {metrics}", now_iso)
    save_run(action="svm", status="Ready", metrics=metrics)

    st.session_state.page = "Results"


def run_train_svm_group_cv():
    if not require_dataset("svm_group_cv"):
        return

    set_status("Running")
    st.session_state.last_action = "svm_group_cv"
    log("Running SVM Group CV started.", now_iso)

    metrics, cm_window, cm_subject, sample_predictions_df, err = train_svm_group_cv(
        st.session_state.dataset_df,
        n_splits=5,
        random_state=42
    )

    if err:
        set_status("Error")
        log(f"SVM Group CV failed: {err}", now_iso)
        st.session_state.last_metrics = {"error": err}
        st.session_state.last_cm = None
        st.session_state.last_cm_window = None
        st.session_state.last_cm_subject = None
        st.session_state.last_roc = None
        st.session_state.last_model = None
        st.session_state.last_group_cv_predictions = None
        save_run(action="svm_group_cv", status="Error", metrics={"error": err})
        return

    st.session_state.last_metrics = metrics
    st.session_state.last_cm = cm_subject
    st.session_state.last_cm_window = cm_window
    st.session_state.last_cm_subject = cm_subject
    st.session_state.last_roc = None
    st.session_state.last_model = None
    st.session_state.last_group_cv_predictions = sample_predictions_df
    st.session_state.last_action = "svm_group_cv"

    set_status("Ready")
    log(
        f"SVM Group CV done. Subject Accuracy mean={metrics.get('subject_acc_mean', 0):.4f}, "
        f"Subject F1 mean={metrics.get('subject_f1_mean', 0):.4f}",
        now_iso
    )
    save_run(action="svm_group_cv", status="Ready", metrics=metrics)

    st.session_state.page = "Results"


def run_train_cnn():
    if not require_dataset("cnn"):
        return

    set_status("Running")
    st.session_state.last_action = "cnn"
    log("Training CNN started (demo mode).", now_iso)
    time.sleep(0.35)

    metrics, cm, roc = fake_cnn_result()

    st.session_state.last_metrics = metrics
    st.session_state.last_cm = cm
    st.session_state.last_cm_window = None
    st.session_state.last_cm_subject = None
    st.session_state.last_roc = roc

    set_status("Ready")
    log(f"CNN done. Metrics: {metrics}", now_iso)
    save_run(action="cnn", status="Ready", metrics=metrics)
    st.session_state.page = "Results"