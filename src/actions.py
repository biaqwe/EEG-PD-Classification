import streamlit as st

from src.ml_models import train_svm, train_svm_group_cv, train_raw_eeg_cnn
from src.cnn_group_cv import train_raw_eeg_cnn_group_cv
from src.state import set_status, log
from src.storage import save_run
from src.utils import now_iso


def require_tabular_dataset(action_name: str): # checks if .csv is loaded
    if st.session_state.dataset_df is None:
        set_status("Error")
        log(f"Cannot run {action_name}: dataset not loaded.", now_iso)
        save_run(action=action_name, status="Error", metrics={"error": "dataset not loaded"})
        return False
    return True


def require_raw_eeg(action_name: str): # checks if raw eeg files are loaded
    if not st.session_state.raw_file_payloads:
        set_status("Error")
        log(f"Cannot run {action_name}: raw EEG files not loaded.", now_iso)
        save_run(action=action_name, status="Error", metrics={"error": "raw EEG files not loaded"})
        return False
    return True


def run_train_svm():
    if not require_tabular_dataset("svm"):
        return

    set_status("Running")
    st.session_state.last_action = "svm"
    log("Training SVM started.", now_iso)

    # calls svm training funct using loaded data
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

    # saves trained model
    st.session_state.last_model = model
    # clears other prediction results
    st.session_state.last_group_cv_predictions = None
    st.session_state.raw_cnn_predictions = None

    # save evaluation results
    st.session_state.last_metrics = metrics
    st.session_state.last_cm = cm
    st.session_state.last_cm_window = None
    st.session_state.last_cm_subject = None
    st.session_state.last_roc = roc

    set_status("Ready")
    log(f"SVM done. Metrics: {metrics}", now_iso)
    save_run(action="svm", status="Ready", metrics=metrics)
    # move to results page
    st.session_state.page = "Results"


def run_train_svm_group_cv():
    if not require_tabular_dataset("svm_group_cv"):
        return

    set_status("Running")
    st.session_state.last_action = "svm_group_cv"
    log("Running SVM Group CV started.", now_iso)

    # calls svm group cv training func using loaded data
    metrics, cm_window, cm_subject, sample_predictions_df, err = train_svm_group_cv(
        st.session_state.dataset_df,
        # data is split into 5 folds
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
    st.session_state.raw_cnn_predictions = None
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
    if not require_raw_eeg("cnn_spectrogram"):
        return

    set_status("Running")
    st.session_state.last_action = "cnn"
    log("Training spectrogram CNN started.", now_iso)

    # calls cnn training func using loaded data
    metrics, cm, roc, model, pred_df, err = train_raw_eeg_cnn(
        payloads=st.session_state.raw_file_payloads,
        config=st.session_state.preprocessing_summary,
    )

    if err:
        set_status("Error")
        log(f"Spectrogram CNN failed: {err}", now_iso)
        st.session_state.last_metrics = {"error": err}
        st.session_state.last_cm = None
        st.session_state.last_cm_window = None
        st.session_state.last_cm_subject = None
        st.session_state.last_roc = None
        st.session_state.last_model = None
        st.session_state.raw_cnn_predictions = None
        save_run(action="cnn_spectrogram", status="Error", metrics={"error": err})
        return

    st.session_state.last_model = model
    st.session_state.last_group_cv_predictions = None
    st.session_state.raw_cnn_predictions = pred_df

    st.session_state.last_metrics = metrics
    st.session_state.last_cm = cm
    st.session_state.last_cm_window = None
    st.session_state.last_cm_subject = None
    st.session_state.last_roc = roc

    set_status("Ready")
    log(f"Spectrogram CNN done. Metrics: {metrics}", now_iso)
    save_run(action="cnn_spectrogram", status="Ready", metrics=metrics)
    st.session_state.page = "Results"


def run_train_cnn_group_cv():
    if not require_raw_eeg("cnn_spectrogram_group_cv"):
        return

    set_status("Running")
    st.session_state.last_action = "cnn_group_cv"
    log("Running spectrogram CNN Group CV started.", now_iso)

    # calls cnn group cv training func using loaded data
    metrics, cm_window, cm_subject, sample_predictions_df, err = train_raw_eeg_cnn_group_cv(
        payloads=st.session_state.raw_file_payloads,
        config=st.session_state.preprocessing_summary,
        n_splits=5,
        random_state=42,
    )

    if err:
        set_status("Error")
        log(f"Spectrogram CNN Group CV failed: {err}", now_iso)
        st.session_state.last_metrics = {"error": err}
        st.session_state.last_cm = None
        st.session_state.last_cm_window = None
        st.session_state.last_cm_subject = None
        st.session_state.last_roc = None
        st.session_state.last_model = None
        st.session_state.raw_cnn_predictions = None
        st.session_state.last_group_cv_predictions = None
        save_run(action="cnn_spectrogram_group_cv", status="Error", metrics={"error": err})
        return

    st.session_state.last_metrics = metrics
    st.session_state.last_cm = cm_subject
    st.session_state.last_cm_window = cm_window
    st.session_state.last_cm_subject = cm_subject
    st.session_state.last_roc = None
    st.session_state.last_model = None
    st.session_state.raw_cnn_predictions = None
    st.session_state.last_group_cv_predictions = sample_predictions_df
    st.session_state.last_action = "cnn_group_cv"

    set_status("Ready")
    log(
        f"Spectrogram CNN Group CV done. Subject Accuracy mean={metrics.get('subject_acc_mean', 0):.4f}, "
        f"Subject F1 mean={metrics.get('subject_f1_mean', 0):.4f}",
        now_iso
    )
    save_run(action="cnn_spectrogram_group_cv", status="Ready", metrics=metrics)
    st.session_state.page = "Results"