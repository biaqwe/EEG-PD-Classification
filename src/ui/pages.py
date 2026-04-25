import json
import io

import pandas as pd
import streamlit as st

from src.actions import (
    run_train_cnn,
    run_train_svm,
    run_train_svm_group_cv,
)
from src.config import (
    RAW_WINDOW_SEC,
    RAW_STEP_SEC,
    RAW_MAX_WINDOWS_PER_RECORDING,
    RAW_L_FREQ,
    RAW_H_FREQ,
    RAW_NOTCH_FREQ,
    RAW_USE_BANDPASS,
    RAW_USE_NOTCH,
    SPECTROGRAM_HEIGHT,
    SPECTROGRAM_WIDTH,
)
from src.data_utils import dataset_summary, parse_csv, parse_iowa_mat
from src.raw_eeg import build_brainvision_payload, load_brainvision_recording_preview
from src.state import log, set_status
from src.storage import load_runs, save_run
from src.ui.components import plot_cm, plot_roc, status_dot
from src.utils import now_iso


def render_dashboard():
    # checks what data is loaded in the app
    df = st.session_state.dataset_df
    raw_ok = st.session_state.raw_file_payloads is not None
    tabular_ok = df is not None

    # calculates nr of rows and feats for .csv files
    csv_rows = "-"
    csv_features = "-"
    if df is not None:
        n_rows, n_features = dataset_summary(df)
        csv_rows = str(n_rows)
        csv_features = str(n_features)

    # calculates nr of recordings and subjects for raw eeg files
    raw_recordings = "-"
    raw_subjects = "-"
    if st.session_state.raw_dataset_summary is not None:
        raw_recordings = str(st.session_state.raw_dataset_summary.get("n_recordings", "-"))
        raw_subjects = str(st.session_state.raw_dataset_summary.get("n_subjects", "-"))

    # splits dashboard in 2 cols
    left, right = st.columns([1.25, 0.75], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Overview</div>
              </div>
              <div class="subtle">
                Use raw BrainVision EEG for CNN and CSV / MAT-based tabular data for SVM baselines.
              </div>
              <div style="height:10px;"></div>
              <div class="kpis">
                <div class="kpi">
                  <div class="lbl">SVM dataset</div>
                  <div class="val">{st.session_state.dataset_name or "Not loaded"}</div>
                  <div class="hint">Used by SVM and SVM Group CV</div>
                </div>
                <div class="kpi">
                  <div class="lbl">SVM rows</div>
                  <div class="val">{csv_rows}</div>
                  <div class="hint">Tabular samples</div>
                </div>
                <div class="kpi">
                  <div class="lbl">Raw recordings</div>
                  <div class="val">{raw_recordings}</div>
                  <div class="hint">Used by raw EEG CNN</div>
                </div>
                <div class="kpi">
                  <div class="lbl">Raw subjects</div>
                  <div class="val">{raw_subjects}</div>
                  <div class="hint">Detected BrainVision subjects</div>
                </div>
              </div>
              <div style="height:10px;"></div>
              <div class="pill">Status: {st.session_state.run_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Recent activity</div>
                <div class="subtle">Local traceability (runs/)</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        # loads last 8 runs
        runs = load_runs(limit=8)
        if runs:
            df_runs = pd.DataFrame(runs)
            cols = ["timestamp", "action", "status", "dataset_name"]
            cols = [c for c in cols if c in df_runs.columns]
            # displays runs as table
            st.dataframe(df_runs[cols], use_container_width=True, hide_index=True)
        else:
            st.info("No runs saved yet.")

    with right:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Quick actions</div>
                <div class="subtle">Start training from here</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        # quick action buttons
        with c1:
            if st.button("Train CNN", use_container_width=True, disabled=not raw_ok):
                run_train_cnn()
        with c2:
            if st.button("Train SVM", use_container_width=True, disabled=not tabular_ok):
                run_train_svm()

        if st.button("Run SVM Group CV", use_container_width=True, disabled=not tabular_ok):
            run_train_svm_group_cv()

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        if st.button("Open Raw EEG Viewer", use_container_width=True, disabled=not raw_ok):
            st.session_state.page = "Raw EEG Viewer"
            st.rerun()

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Logs</div>
                <div class="subtle">Execution messages</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        # collects last 200 logs
        logs_text = "\n".join(st.session_state.logs[-200:]) if st.session_state.logs else "No logs yet."
        st.markdown(
            f"<div class='logbox'>{logs_text.replace('<','&lt;').replace('>','&gt;')}</div>",
            unsafe_allow_html=True
        )


def render_import():
    st.markdown(
        """
        <div class="card">
          <div class="card-title">
            <div style="font-weight:800; font-size:1.1rem;">Import datasets</div>
            <div class="subtle">Use one import flow for raw EEG CNN and a separate one for CSV or Iowa .mat based SVM models</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="large")

    # checks if raw eeg data was uploaded and calculates nr of recordings, subjects and complete triplets (complete triplets means .eeg .vhdr .vmrk for the same subject)
    with col_left:
        current_raw_manifest_df = st.session_state.raw_manifest_df
        raw_loaded = current_raw_manifest_df is not None
        raw_recordings = int(current_raw_manifest_df["recording"].nunique()) if raw_loaded else 0
        raw_subjects = int(current_raw_manifest_df["subject_key"].nunique()) if raw_loaded else 0
        raw_complete = int(current_raw_manifest_df["complete_triplet"].sum()) if raw_loaded else 0

        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.0rem;">Raw EEG import for CNN</div>
                <div class="subtle">BrainVision files: .vhdr, .eeg, .vmrk</div>
              </div>
              <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
                <span class="pill">Status: <b style="color:var(--txt)">{'Loaded' if raw_loaded else 'Not loaded'}</b></span>
                <span class="pill">Recordings: <b style="color:var(--txt)">{raw_recordings}</b></span>
                <span class="pill">Subjects: <b style="color:var(--txt)">{raw_subjects}</b></span>
                <span class="pill">Complete triplets: <b style="color:var(--txt)">{raw_complete}</b></span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # for file upload
        uploaded_raw_files = st.file_uploader(
            "Upload BrainVision files",
            type=["vhdr", "eeg", "vmrk"],
            accept_multiple_files=True,
            key="raw_eeg_uploader",
            help="Select all .vhdr, .eeg and .vmrk files used by the raw EEG CNN.",
        )

        raw_name = st.text_input(
            "Raw EEG dataset name",
            value="raw_eeg",
            key="raw_dataset_name_input",
        )

        st.caption("Upload complete file triplets for each recording with matching base names.")

        if st.button(
            "Load raw EEG dataset",
            use_container_width=True,
            disabled=not uploaded_raw_files,
            key="load_raw_eeg_btn",
        ): #uploaded files are analyzed
            payloads, manifest_df, err = build_brainvision_payload(uploaded_raw_files)

            if err:
                set_status("Error")
                log(f"Raw EEG import failed: {err}", now_iso)
                st.error(err)
            else:
                st.session_state.raw_file_payloads = payloads
                st.session_state.raw_dataset_name = raw_name.strip() or "brainvision_raw_eeg"
                st.session_state.raw_manifest_df = manifest_df
                st.session_state.raw_cnn_predictions = None

                summary = {
                    "n_recordings": int(manifest_df["recording"].nunique()),
                    "n_subjects": int(manifest_df["subject_key"].nunique()),
                    "n_channels": None,
                }
                st.session_state.raw_dataset_summary = summary

                set_status("Ready")
                log(
                    f"Raw BrainVision dataset loaded: {raw_name.strip() or 'brainvision_raw_eeg'} "
                    f"({summary['n_recordings']} recordings)",
                    now_iso
                )
                save_run(
                    action="import_raw_brainvision",
                    status="Ready",
                    metrics={
                        "n_recordings": summary["n_recordings"],
                        "n_subjects": summary["n_subjects"],
                    },
                )
                st.success("Raw EEG dataset loaded successfully.")
                st.rerun()

        # data preview
        if st.session_state.raw_manifest_df is not None:
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            with st.expander("Preview detected BrainVision recordings", expanded=True):
                st.dataframe(st.session_state.raw_manifest_df, use_container_width=True, hide_index=True)

            if st.button("Open Raw EEG Viewer", use_container_width=True, key="open_raw_viewer_from_import"):
                st.session_state.page = "Raw EEG Viewer"
                st.rerun()

            incomplete_df = st.session_state.raw_manifest_df[
                ~st.session_state.raw_manifest_df["complete_triplet"]
            ]
            if not incomplete_df.empty:
                st.warning("Some recordings are incomplete and will not be usable.")
                st.dataframe(incomplete_df, use_container_width=True, hide_index=True)

    # checks if svm data was uploaded and calculates nr of rows and feats
    with col_right:
        current_svm_df = st.session_state.dataset_df
        svm_loaded = current_svm_df is not None
        svm_rows = 0
        svm_features = 0
        if svm_loaded:
            svm_rows, svm_features = dataset_summary(current_svm_df)

        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.0rem;">SVM dataset import</div>
                <div class="subtle">Upload either a ready CSV or an Iowa .mat file</div>
              </div>
              <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
                <span class="pill">Status: <b style="color:var(--txt)">{'Loaded' if svm_loaded else 'Not loaded'}</b></span>
                <span class="pill">Rows: <b style="color:var(--txt)">{svm_rows}</b></span>
                <span class="pill">Features: <b style="color:var(--txt)">{svm_features}</b></span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # for .csv upload
        uploaded_csv = st.file_uploader(
            "Upload CSV dataset",
            type=["csv"],
            accept_multiple_files=False,
            key="csv_uploader",
            help="Ready-made tabular dataset for SVM.",
        )

        # for .mat upload
        uploaded_mat = st.file_uploader(
            "Upload Iowa .mat dataset",
            type=["mat"],
            accept_multiple_files=False,
            key="mat_uploader",
            help="Will be converted to tabular SVM features using the saved preprocessing config.",
        )

        svm_name = st.text_input(
            "SVM dataset name",
            value=st.session_state.dataset_name or "eeg_csv",
            key="csv_dataset_name_input",
        )

        can_load_svm = (uploaded_csv is not None) or (uploaded_mat is not None)

        if st.button(
            "Load SVM dataset",
            use_container_width=True,
            disabled=not can_load_svm,
            key="load_svm_btn",
        ): # uploaded files are analyzed
            try:
                if uploaded_csv is not None and uploaded_mat is not None:
                    st.error("Please upload only one source at a time: either CSV or MAT.")
                elif uploaded_csv is not None:
                    df = parse_csv(uploaded_csv)
                    if df is None:
                        raise ValueError("Could not parse CSV file.")

                    st.session_state.dataset_df = df
                    st.session_state.dataset_name = svm_name.strip() or "eeg_csv"
                    st.session_state.dataset_source = "csv"
                    st.session_state.last_group_cv_predictions = None

                    n_rows, n_features = dataset_summary(df)

                    set_status("Ready")
                    log(
                        f"CSV dataset loaded: {st.session_state.dataset_name} "
                        f"({n_rows} rows, {n_features} features)",
                        now_iso
                    )
                    save_run(
                        action="import_csv",
                        status="Ready",
                        metrics={"rows": n_rows, "features": n_features},
                    )
                    st.success("CSV dataset loaded successfully.")
                    st.rerun()

                elif uploaded_mat is not None:
                    st.session_state.mat_file_payload = uploaded_mat.getvalue()
                    st.session_state.mat_file_name = uploaded_mat.name

                    df, summary, err = parse_iowa_mat(
                        uploaded_mat,
                        preprocessing_summary=st.session_state.preprocessing_summary,
                    )

                    if err:
                        raise ValueError(err)

                    st.session_state.dataset_df = df
                    st.session_state.dataset_name = svm_name.strip() or uploaded_mat.name.rsplit(".", 1)[0]
                    st.session_state.dataset_source = "mat"
                    st.session_state.last_group_cv_predictions = None

                    n_rows, n_features = dataset_summary(df)

                    set_status("Ready")
                    log(
                        f"MAT dataset converted for SVM: {st.session_state.dataset_name} "
                        f"({n_rows} rows, {n_features} features)",
                        now_iso
                    )
                    save_run(
                        action="import_mat_for_svm",
                        status="Ready",
                        metrics={
                            "rows": n_rows,
                            "features": n_features,
                            "window_sec": summary.get("window_sec") if summary else None,
                            "step_sec": summary.get("step_sec") if summary else None,
                            "valid_channels": summary.get("valid_channels") if summary else None,
                        },
                    )
                    st.success("MAT dataset converted and loaded successfully.")
                    st.rerun()

            except Exception as e:
                set_status("Error")
                log(f"SVM dataset import failed: {e}", now_iso)
                st.error(f"Could not load SVM dataset: {e}")

        # data preview
        if st.session_state.dataset_df is not None:
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            with st.expander("Preview SVM dataset rows", expanded=True):
                st.dataframe(st.session_state.dataset_df.head(20), use_container_width=True, hide_index=True)

            st.caption(
                "If you upload a .mat file, the saved preprocessing config is used to generate the tabular dataset."
            )

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
          <div class="card-title">
            <div style="font-weight:800; font-size:1.0rem;">How to use this page</div>
            <div class="subtle">Choose the right import depending on the model</div>
          </div>
          <div class="small" style="margin-top:8px;">
            - Use <b>Raw EEG import</b> for <b>Train CNN</b><br/>
            - Use <b>CSV import</b> if you already have a ready tabular dataset for <b>SVM</b><br/>
            - Use <b>Iowa .mat import</b> if you want the app to generate the SVM dataset using the saved preprocessing config
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_preprocess():
    st.markdown(
        """
        <div class="card">
          <div class="card-title">
            <div style="font-weight:800; font-size:1.05rem;">Preprocessing config</div>
            <div class="subtle">Windowing + filtering settings used by Raw EEG CNN and Iowa .mat → SVM conversion</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.0, 1.0], gap="large")

    # loads default config
    default_cfg = st.session_state.preprocessing_summary or {
        "window_sec": float(RAW_WINDOW_SEC),
        "step_sec": float(RAW_STEP_SEC),
        "max_windows_per_recording": int(RAW_MAX_WINDOWS_PER_RECORDING),
        "use_notch": bool(RAW_USE_NOTCH),
        "notch_freq": float(RAW_NOTCH_FREQ),
        "use_bandpass": bool(RAW_USE_BANDPASS),
        "bandpass_low": float(RAW_L_FREQ),
        "bandpass_high": float(RAW_H_FREQ),
        "visual_height": int(SPECTROGRAM_HEIGHT),
        "visual_width": int(SPECTROGRAM_WIDTH),
    }

    with left:
        c1, c2 = st.columns(2)
        with c1:
            window_sec = st.number_input(
                "Window length (sec)",
                min_value=0.5,
                value=float(default_cfg["window_sec"]),
                step=0.5
            )
        with c2:
            step_sec = st.number_input(
                "Step (sec)",
                min_value=0.5,
                value=float(default_cfg["step_sec"]),
                step=0.5
            )

        c3, c4 = st.columns(2)
        with c3:
            max_windows = st.number_input(
                "Max windows / recording",
                min_value=1,
                value=int(default_cfg["max_windows_per_recording"]),
                step=1
            )
        with c4:
            notch_freq = st.number_input(
                "Notch (Hz)",
                min_value=0.0,
                value=float(default_cfg["notch_freq"]),
                step=0.5
            )

        use_notch = st.checkbox("Use notch filter", value=bool(default_cfg["use_notch"]))
        use_bandpass = st.checkbox("Use band-pass filter", value=bool(default_cfg["use_bandpass"]))

        if use_bandpass:
            c5, c6 = st.columns(2)
            with c5:
                bandpass_low = st.number_input(
                    "Band-pass low (Hz)",
                    min_value=0.0,
                    value=float(default_cfg["bandpass_low"]),
                    step=0.1
                )
            with c6:
                bandpass_high = st.number_input(
                    "Band-pass high (Hz)",
                    min_value=0.1,
                    value=float(default_cfg["bandpass_high"]),
                    step=0.5
                )
        else:
            bandpass_low = float(default_cfg["bandpass_low"])
            bandpass_high = float(default_cfg["bandpass_high"])

        c7, c8 = st.columns(2)
        with c7:
            visual_height = st.number_input(
                "Spectrogram height",
                min_value=32,
                max_value=256,
                value=int(default_cfg.get("visual_height", SPECTROGRAM_HEIGHT)),
                step=16
            )
        with c8:
            visual_width = st.number_input(
                "Spectrogram width",
                min_value=32,
                max_value=256,
                value=int(default_cfg.get("visual_width", SPECTROGRAM_WIDTH)),
                step=16
            )

        # saves current config settings
        if st.button("Save preprocessing config", use_container_width=True):
            new_summary = {
                "window_sec": float(window_sec),
                "step_sec": float(step_sec),
                "max_windows_per_recording": int(max_windows),
                "use_notch": bool(use_notch),
                "notch_freq": float(notch_freq),
                "use_bandpass": bool(use_bandpass),
                "bandpass_low": float(bandpass_low),
                "bandpass_high": float(bandpass_high),
                "visual_height": int(visual_height),
                "visual_width": int(visual_width),
            }

            config_changed = st.session_state.preprocessing_summary != new_summary

            st.session_state.preprocessing_summary = new_summary
            st.session_state.preprocessing_logs = [
                f"Window length: {window_sec} sec",
                f"Step: {step_sec} sec",
                f"Max windows / recording: {max_windows}",
                f"Use notch: {use_notch}",
                f"Notch freq: {notch_freq} Hz",
                f"Use bandpass: {use_bandpass}",
                f"Bandpass: {bandpass_low} - {bandpass_high} Hz",
                f"Spectrogram size: {visual_height} x {visual_width}",
            ]

            if config_changed:
                st.session_state.raw_cnn_predictions = None
                st.session_state.last_model = None
                st.session_state.last_metrics = {}
                st.session_state.last_cm = None
                st.session_state.last_roc = None

                if st.session_state.dataset_source == "mat" and st.session_state.mat_file_payload is not None:
                    mat_buffer = io.BytesIO(st.session_state.mat_file_payload)
                    mat_buffer.name = st.session_state.mat_file_name or "dataset.mat"
                    df, summary, err = parse_iowa_mat(
                        mat_buffer,
                        preprocessing_summary=new_summary,
                    )

                    if err:
                        st.warning(f"Config was saved, but the stored MAT dataset could not be regenerated: {err}")
                    else:
                        st.session_state.dataset_df = df
                        st.session_state.last_group_cv_predictions = None
                        n_rows, n_features = dataset_summary(df)
                        log(
                            f"Stored MAT dataset regenerated with new preprocessing config "
                            f"({n_rows} rows, {n_features} features).",
                            now_iso
                        )

                elif st.session_state.dataset_source == "mat":
                    st.session_state.dataset_df = None
                    st.session_state.dataset_name = None
                    st.session_state.dataset_source = None
                    st.session_state.last_group_cv_predictions = None

            if (
                st.session_state.dataset_df is not None
                or st.session_state.raw_file_payloads is not None
            ):
                set_status("Ready")
            else:
                set_status("Idle")

            log("Raw EEG preprocessing config updated.", now_iso)
            st.success("Configuration saved.")
            st.rerun()

    with right:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Current config</div>
                <div class="subtle">Used by Train CNN and by MAT → SVM feature generation</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        # shows saved config
        if st.session_state.preprocessing_summary is None:
            st.info("No preprocessing config saved yet.")
        else:
            st.json(st.session_state.preprocessing_summary)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Notes</div>
                <div class="subtle">What this page does</div>
              </div>
              <div class="small">
                - Used by <b>Train CNN</b><br/>
                - Also used when converting <b>Iowa .mat</b> into a tabular dataset for <b>SVM</b><br/>
                - Applies optional notch and band-pass filtering<br/>
                - Segments data into fixed windows<br/>
                - Save configuration does not process data yet<br/>
                - The actual MAT conversion happens when you load the .mat file in Import
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # shows preprocessing logs if any
        if st.session_state.preprocessing_logs:
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            logs_text = "\n".join(st.session_state.preprocessing_logs)
            st.markdown(
                f"<div class='logbox'>{logs_text.replace('<','&lt;').replace('>','&gt;')}</div>",
                unsafe_allow_html=True
            )


def get_last_model_display_name(): # returns the name of the last trained model
    action = st.session_state.last_action

    mapping = {
        "cnn": "CNN",
        "svm": "SVM",
        "svm_group_cv": "SVM Group CV",
        "cnn_raw_eeg": "CNN",
    }

    return mapping.get(action, "Unknown model")


def render_results():
    model_name = get_last_model_display_name()

    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">
            <div>
              <div style="font-weight:800; font-size:1.05rem;">Evaluation and model comparison</div>
              <div class="subtle">Metrics, confusion matrix, ROC/AUC, export</div>
            </div>
            <div class="pill">Model: {model_name}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

    # gets the metrics from last train
    metrics = st.session_state.last_metrics or {}

    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">
            <div>
              <div style="font-weight:800; font-size:1.05rem;">Metrics</div>
              <div class="subtle">Last run</div>
            </div>
            <div class="pill">Model: {model_name}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

    if not metrics:
        st.info("No metrics yet. Train a model first.")
    else:
        # svm group cv metrics
        if st.session_state.last_action == "svm_group_cv":
            st.markdown("**Subject-level mean performance (Group CV)**")
            mcols = st.columns(3)

            acc = metrics.get("subject_acc_mean", None)
            f1 = metrics.get("subject_f1_mean", None)
            auc = metrics.get("subject_auc_mean", None)

            with mcols[0]:
                st.metric("Subject Accuracy", "-" if acc is None else f"{acc:.3f}")
            with mcols[1]:
                st.metric("Subject F1", "-" if f1 is None else f"{f1:.3f}")
            with mcols[2]:
                st.metric("Subject AUC", "-" if auc is None else f"{auc:.3f}")
        else:
            # cnn and svm common metrivcs
            mcols = st.columns(3)

            acc = metrics.get("accuracy", None)
            f1 = metrics.get("f1", None)
            auc = metrics.get("auc", None)

            with mcols[0]:
                st.metric("Accuracy", "-" if acc is None else f"{acc:.3f}")
            with mcols[1]:
                st.metric("F1", "-" if f1 is None else f"{f1:.3f}")
            with mcols[2]:
                st.metric("AUC", "-" if auc is None else f"{auc:.3f}")

        if st.session_state.last_action == "cnn":
            st.markdown("**Subject-level performance (averaged window probabilities)**")
            subject_cols = st.columns(3)
            subject_acc = metrics.get("subject_accuracy", None)
            subject_f1 = metrics.get("subject_f1", None)
            subject_auc = metrics.get("subject_auc", None)

            with subject_cols[0]:
                st.metric("Subject Accuracy", "-" if subject_acc is None else f"{subject_acc:.3f}")
            with subject_cols[1]:
                st.metric("Subject F1", "-" if subject_f1 is None else f"{subject_f1:.3f}")
            with subject_cols[2]:
                st.metric("Subject AUC", "-" if subject_auc is None else f"{subject_auc:.3f}")

            st.markdown("**CNN diagnostic checks**")
            diag_cols = st.columns(4)

            with diag_cols[0]:
                v = metrics.get("window_mean_proba_pd_for_true_hc", None)
                st.metric("Mean PD prob | true HC", "-" if v is None else f"{v:.3f}")

            with diag_cols[1]:
                v = metrics.get("window_mean_proba_pd_for_true_pd", None)
                st.metric("Mean PD prob | true PD", "-" if v is None else f"{v:.3f}")

            with diag_cols[2]:
                v = metrics.get("val_subject_auc", None)
                st.metric("Val subject AUC", "-" if v is None else f"{v:.3f}")

            with diag_cols[3]:
                v = metrics.get("subject_auc_if_flipped", None)
                st.metric("Subject AUC if flipped", "-" if v is None else f"{v:.3f}")

            if metrics.get("subject_looks_inverted", False):
                st.warning(
                    "The model gives higher PD probabilities to HC subjects than to PD subjects. "
                    "Check label detection in the raw EEG manifest and inspect filenames."
                )

            st.markdown("**Subject split check**")

            leakage_detected = metrics.get("subject_leakage_detected", None)

            if leakage_detected is False:
                st.success("No subject leakage detected between train, validation and test splits.")
            elif leakage_detected is True:
                st.error("Subject leakage detected.")
            else:
                st.info("Subject leakage check not available for this run.")

            split_rows = [
                {
                    "split": "train",
                    "subjects": metrics.get("train_subjects", "-"),
                    "HC subjects": metrics.get("train_hc_subjects", "-"),
                    "PD subjects": metrics.get("train_pd_subjects", "-"),
                    "windows": metrics.get("train_windows", "-"),
                    "HC windows": metrics.get("train_hc_windows", "-"),
                    "PD windows": metrics.get("train_pd_windows", "-"),
                    "subject keys": metrics.get("train_subject_keys", "-"),
                },
                {
                    "split": "validation",
                    "subjects": metrics.get("val_subjects", "-"),
                    "HC subjects": metrics.get("val_hc_subjects", "-"),
                    "PD subjects": metrics.get("val_pd_subjects", "-"),
                    "windows": metrics.get("val_windows", "-"),
                    "HC windows": metrics.get("val_hc_windows", "-"),
                    "PD windows": metrics.get("val_pd_windows", "-"),
                    "subject keys": metrics.get("val_subject_keys", "-"),
                },
                {
                    "split": "test",
                    "subjects": metrics.get("test_subjects", "-"),
                    "HC subjects": metrics.get("test_hc_subjects", "-"),
                    "PD subjects": metrics.get("test_pd_subjects", "-"),
                    "windows": metrics.get("test_windows", "-"),
                    "HC windows": metrics.get("test_hc_windows", "-"),
                    "PD windows": metrics.get("test_pd_windows", "-"),
                    "subject keys": metrics.get("test_subject_keys", "-"),
                },
            ]

            st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)

        if "error" in metrics:
            st.error(metrics["error"])

        # extra info for cnn
        if st.session_state.last_action == "cnn":
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            extra_cols = st.columns(6)
            with extra_cols[0]:
                st.metric("Subjects", str(metrics.get("n_subjects", "-")))
            with extra_cols[1]:
                st.metric("Recordings", str(metrics.get("n_recordings", "-")))
            with extra_cols[2]:
                st.metric("Raw channels", str(metrics.get("n_channels", "-")))
            with extra_cols[3]:
                st.metric("CNN input channels", str(metrics.get("cnn_input_channels", "-")))
            with extra_cols[4]:
                st.metric("Spectrogram", f"{metrics.get('visual_height', '-')} x {metrics.get('visual_width', '-')}")
            with extra_cols[5]:
                threshold = metrics.get("threshold", None)
                st.metric("Threshold", "-" if threshold is None else f"{threshold:.3f}")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
          <div class="card-title">
            <div style="font-weight:800; font-size:1.05rem;">Visualizations</div>
            <div class="subtle">Confusion matrix and ROC</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

    # two cms for group cv, one subject level and one window level
    if st.session_state.last_action == "svm_group_cv":
        viz1, viz2 = st.columns(2, gap="large")

        with viz1:
            if st.session_state.last_cm_subject is not None:
                st.markdown("**Subject-level confusion matrix**")
                plot_cm(st.session_state.last_cm_subject, title="Subject-level Confusion Matrix")
            else:
                st.info("Subject-level confusion matrix not available yet.")

        with viz2:
            if st.session_state.last_cm_window is not None:
                st.markdown("**Window-level confusion matrix**")
                plot_cm(st.session_state.last_cm_window, title="Window-level Confusion Matrix")
            else:
                st.info("Window-level confusion matrix not available yet.")
    # cm and roc for cnn and svm
    else:
        viz1, viz2 = st.columns(2, gap="large")

        with viz1:
            if st.session_state.last_cm is not None:
                plot_cm(st.session_state.last_cm)
            else:
                st.info("Confusion matrix not available yet.")

        with viz2:
            if st.session_state.last_roc is not None:
                plot_roc(st.session_state.last_roc)
            else:
                st.info("ROC curve not available yet.")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
          <div class="card-title">
            <div style="font-weight:800; font-size:1.05rem;">Sample prediction</div>
            <div class="subtle">Inspect one sample prediction</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

    # sample prediction for cnn
    if st.session_state.last_action == "cnn":
        pred_df = st.session_state.raw_cnn_predictions
        if pred_df is None or pred_df.empty:
            st.info("Sample prediction is available after Train CNN.")
        else:
            sample_idx = st.number_input(
                "Select CNN test sample index",
                min_value=0,
                max_value=len(pred_df) - 1,
                value=0,
                step=1,
                key="sample_prediction_index_cnn",
            )

            row = pred_df.iloc[int(sample_idx)]
            # convers labels to text
            true_label = "PD" if int(row["true_label"]) == 1 else "HC"
            pred_label = "PD" if int(row["pred_label"]) == 1 else "HC"
            # compute confidenec and correctness
            confidence = float(max(row["proba_pd"], row["proba_hc"]))
            correct = int(row["true_label"]) == int(row["pred_label"])
            # displays info about selected samplle
            info_df = pd.DataFrame([{
                "recording": row.get("recording", ""),
                "subject_key": row.get("subject_key", ""),
                "window_index": int(row.get("window_index", -1)),
                "start_sample": int(row.get("start_sample", -1)),
                "true_label": true_label,
            }])

            st.dataframe(info_df, use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("True label", true_label)
            with c2:
                st.metric("Predicted label", pred_label)
            with c3:
                st.metric("Confidence", f"{confidence:.3f}")

            c4, c5 = st.columns(2)
            with c4:
                st.metric("Probability PD", f"{float(row['proba_pd']):.3f}")
            with c5:
                st.metric("Probability HC", f"{float(row['proba_hc']):.3f}")

            if correct:
                st.success("Correct prediction")
            else:
                st.error("Incorrect prediction")

    # sample prediction for svm
    elif st.session_state.last_action == "svm":
        df = st.session_state.dataset_df
        model = st.session_state.last_model

        if df is None or model is None:
            st.info("Sample prediction is available after Train SVM.")
        else:
            try:
                # finds which col contains true class labels
                label_candidates = [c for c in df.columns if c.lower() in ["label", "class", "y", "target"]]
                if not label_candidates:
                    st.warning("No label column found in dataset.")
                else:
                    ycol = label_candidates[0]
                    sample_idx = st.number_input(
                        "Select sample index",
                        min_value=0,
                        max_value=len(df) - 1,
                        value=0,
                        step=1,
                        key="sample_prediction_index_svm",
                    )

                    sample = df.iloc[[sample_idx]].copy()
                    y_true = sample[ycol].iloc[0]
                    # removes cols that shouldnt be used as model feats
                    meta_cols = [
                        c for c in [
                            "group", "subject_id", "subject_key", "window_start",
                            "recording", "part", "start", "source_file",
                        ] if c in sample.columns
                    ]

                    X_sample = sample.drop(columns=[ycol] + meta_cols, errors="ignore")

                    # run svm prediction
                    pred = model.predict(X_sample)[0]
                    proba = model.predict_proba(X_sample)[0]

                    pred_label = "PD" if int(pred) == 1 else "HC"
                    true_label = "PD" if int(y_true) == 1 else "HC"

                    confidence = float(max(proba))
                    correct = int(pred) == int(y_true)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("True label", true_label)
                    with c2:
                        st.metric("Predicted label", pred_label)
                    with c3:
                        st.metric("Confidence", f"{confidence:.3f}")

                    c4, c5 = st.columns(2)
                    with c4:
                        st.metric("Probability PD", f"{proba[1]:.3f}")
                    with c5:
                        st.metric("Probability HC", f"{proba[0]:.3f}")

                    if correct:
                        st.success("Correct prediction")
                    else:
                        st.error("Incorrect prediction")

            except Exception as e:
                st.error(f"Could not generate sample prediction: {e}")

    # sample prediction for svm group cv
    elif st.session_state.last_action == "svm_group_cv":
        pred_df = st.session_state.last_group_cv_predictions

        if pred_df is None or pred_df.empty:
            st.info("Sample prediction is available after Run SVM Group CV.")
        else:
            sample_idx = st.number_input(
                "Select sample index",
                min_value=0,
                max_value=len(pred_df) - 1,
                value=0,
                step=1,
                key="sample_prediction_index_group_cv",
            )

            row = pred_df.iloc[int(sample_idx)]

            true_label = "PD" if int(row["true_label"]) == 1 else "HC"
            pred_label = "PD" if int(row["pred_label"]) == 1 else "HC"

            confidence = float(max(row["proba_pd"], row["proba_hc"]))
            correct = int(row["true_label"]) == int(row["pred_label"])
            # shows sample metadata
            info_df = pd.DataFrame([{
                "row_index": int(row["row_index"]),
                "fold": int(row["fold"]),
                "subject_id": row.get("subject_id", ""),
                "subject_key": row.get("subject_key", ""),
                "window_start": int(row.get("window_start", -1)),
                "true_label": true_label,
            }])

            st.dataframe(info_df, use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("True label", true_label)
            with c2:
                st.metric("Predicted label", pred_label)
            with c3:
                st.metric("Confidence", f"{confidence:.3f}")

            c4, c5 = st.columns(2)
            with c4:
                st.metric("Probability PD", f"{float(row['proba_pd']):.3f}")
            with c5:
                st.metric("Probability HC", f"{float(row['proba_hc']):.3f}")

            if correct:
                st.success("Correct prediction")
            else:
                st.error("Incorrect prediction")
    else:
        st.info("Train a model first.")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
          <div class="card-title">
            <div style="font-weight:800; font-size:1.05rem;">Export and traceability</div>
            <div class="subtle">Download latest run JSON</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

    # loads last saved run
    runs = load_runs(limit=1)
    if not runs:
        st.info("No run record to export yet.")
    else:
        # shows last run and enables download
        last = runs[0]
        st.json(last, expanded=False)
        payload = json.dumps(last, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "Download last run JSON",
            data=payload,
            file_name=f"run_{last.get('run_id','latest')}.json",
            mime="application/json",
            use_container_width=True,
        )


def render_raw_viewer():
    st.markdown(
        """
        <div class="card">
          <div class="card-title">
            <div>
              <div style="font-weight:800; font-size:1.1rem;">Preprocessed EEG signal viewer</div>
              <div class="subtle">Visual inspection after applying the saved preprocessing configuration</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    if st.session_state.raw_manifest_df is None or not st.session_state.raw_file_payloads: # checks if raw eeg data was loaded
        st.info("Load a raw EEG BrainVision dataset first from the Import page.")
        if st.button("Go to Import", use_container_width=False):
            st.session_state.page = "Import"
            st.rerun()
        return

    manifest_df = st.session_state.raw_manifest_df.copy() # contains info about uploaded recordinggs
    complete_df = manifest_df[manifest_df["complete_triplet"]].copy() # only keeps recordings with a full triplet

    if complete_df.empty:
        st.warning("No complete BrainVision recordings are available for visualization.")
        return

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        # recording selector
        recordings = complete_df["recording"].tolist()
        selected_recording = st.selectbox(
            "Recording",
            options=recordings,
            index=0,
            key="raw_viewer_recording",
        )

        # time range slider
        preview_seconds = st.slider(
            "Seconds to display",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            key="raw_viewer_seconds",
        )

        data, times, summary, err = load_brainvision_recording_preview(
            st.session_state.raw_file_payloads,
            selected_recording,
            seconds=float(preview_seconds),
            config=st.session_state.preprocessing_summary,
        )

        if err:
            st.error(err)
            return

        # reads the list of channel names from the loaded recording
        available_channels = summary["channel_names"]
        default_count = min(6, len(available_channels))
        selected_channels = st.multiselect(
            "Channels",
            options=available_channels,
            default=available_channels[:default_count],
            key="raw_viewer_channels",
        )

        if not selected_channels:
            st.warning("Select at least one channel to display the signal.")
            return

        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.0rem;">Recording summary</div>
                <div class="subtle">{summary['recording']}</div>
              </div>
              <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
                <span class="pill">Channels: <b style="color:var(--txt)">{summary['n_channels']}</b></span>
                <span class="pill">Samples: <b style="color:var(--txt)">{summary['n_samples']}</b></span>
                <span class="pill">Sampling rate: <b style="color:var(--txt)">{summary['sampling_rate']:.2f} Hz</b></span>
                <span class="pill">Displayed duration: <b style="color:var(--txt)">{summary['duration_sec']:.2f} s</b></span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        preview_df = complete_df[["recording", "subject_key", "label_guess"]].copy()
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

    with right:
        import matplotlib.pyplot as plt
        import numpy as np

        channel_to_idx = {name: idx for idx, name in enumerate(available_channels)} # creates a dict to link each channel name to its numeric indesx
        ordered_indices = [channel_to_idx[name] for name in selected_channels] # converts the selected channel names into indices
        selected_data = data[ordered_indices] # extracts only the selected channel sigs from the eeg data

        # finds the largest absolute sig value from the selected channels
        scale = np.max(np.abs(selected_data))
        if scale == 0:
            scale = 1.0

        fig = plt.figure(figsize=(12, 7))
        spacing = 3.0

        for plot_idx, channel_name in enumerate(selected_channels):
            signal = selected_data[plot_idx] / scale # divides by scale
            offset = (len(selected_channels) - 1 - plot_idx) * spacing # vertical offset
            plt.plot(times, signal + offset, linewidth=0.9) # upward or downward shift
            plt.text(times[0] if len(times) else 0, offset, channel_name, va="bottom", ha="left")

        plt.title(f"Preprocessed EEG signals - {selected_recording}")
        plt.xlabel("Time (s)")
        plt.yticks([])
        plt.grid(True, alpha=0.25)
        st.pyplot(fig, clear_figure=True)

        raw_table = {"time_sec": times}
        for channel_name in selected_channels:
            raw_table[channel_name] = data[channel_to_idx[channel_name]]
        raw_table_df = pd.DataFrame(raw_table)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        with st.expander("Preview signal values", expanded=False):
            st.dataframe(raw_table_df.head(300), use_container_width=True, hide_index=True)