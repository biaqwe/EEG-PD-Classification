import json
import io

import pandas as pd
import streamlit as st

from src.actions import (
    run_train_cnn_group_cv,
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
    SPECTROGRAM_NPERSEG,
    SPECTROGRAM_OVERLAP_RATIO,
    SPECTROGRAM_CHANNEL_MODE,
)
from src.data_utils import dataset_summary, get_xy, parse_csv, parse_mat_dataset
from src.raw_eeg import (
    build_brainvision_payload,
    load_brainvision_feature_table,
    load_brainvision_recording_preview,
)
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
                Track the loaded tabular feature table for SVM Group CV and raw EEG recordings for CNN Group CV.
              </div>
              <div style="height:10px;"></div>
              <div class="kpis">
                <div class="kpi">
                  <div class="lbl">Feature table</div>
                  <div class="val">{st.session_state.dataset_name or "Not loaded"}</div>
                  <div class="hint">Input for SVM Group CV</div>
                </div>
                <div class="kpi">
                  <div class="lbl">Tabular samples</div>
                  <div class="val">{csv_rows}</div>
                  <div class="hint">{csv_features} numeric features</div>
                </div>
                <div class="kpi">
                  <div class="lbl">EEG recordings</div>
                  <div class="val">{raw_recordings}</div>
                  <div class="hint">Input for CNN Group CV</div>
                </div>
                <div class="kpi">
                  <div class="lbl">EEG subjects</div>
                  <div class="val">{raw_subjects}</div>
                  <div class="hint">Subject groups for validation</div>
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

        if st.button("Run SVM Group CV", use_container_width=True, disabled=not tabular_ok):
            run_train_svm_group_cv()

        if st.button("Run CNN Group CV", use_container_width=True, disabled=not raw_ok):
            run_train_cnn_group_cv()

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
              <div class="subtle">Use one import flow for raw EEG CNN and a separate one for CSV or MAT-based SVM models</div>
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
        raw_complete = int(current_raw_manifest_df["complete_recording"].sum()) if raw_loaded else 0

        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.0rem;">Raw EEG import for CNN</div>
                <div class="subtle">EEGLAB .set/.fdt files or BrainVision .vhdr/.eeg/.vmrk files</div>
              </div>
              <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
                <span class="pill">Status: <b style="color:var(--txt)">{'Loaded' if raw_loaded else 'Not loaded'}</b></span>
                <span class="pill">Recordings: <b style="color:var(--txt)">{raw_recordings}</b></span>
                <span class="pill">Subjects: <b style="color:var(--txt)">{raw_subjects}</b></span>
                <span class="pill">Complete recordings: <b style="color:var(--txt)">{raw_complete}</b></span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # for file upload
        uploaded_raw_files = st.file_uploader(
            "Upload raw EEG files",
            type=["set", "fdt", "tsv", "csv", "vhdr", "eeg", "vmrk"],
            accept_multiple_files=True,
            key="raw_eeg_uploader",
            help=(
                "For the new dataset, upload all .set and .fdt files plus participants.tsv if available. "
                "BrainVision .vhdr/.eeg/.vmrk files are still supported."
            ),
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
                    f"Raw EEG dataset loaded: {raw_name.strip() or 'raw_eeg'} "
                    f"({summary['n_recordings']} recordings)",
                    now_iso
                )
                save_run(
                    action="import_raw_eeg",
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

            if st.button(
                "Generate SVM features from loaded raw EEG",
                use_container_width=True,
                key="generate_svm_from_raw_eeg",
            ):
                df_features, feature_summary, feature_err = load_brainvision_feature_table(
                    st.session_state.raw_file_payloads,
                    config=st.session_state.preprocessing_summary,
                )

                if feature_err:
                    set_status("Error")
                    log(f"Raw EEG feature generation failed: {feature_err}", now_iso)
                    st.error(feature_err)
                else:
                    st.session_state.dataset_df = df_features
                    st.session_state.dataset_name = f"{st.session_state.raw_dataset_name or 'raw_eeg'}_features"
                    st.session_state.dataset_source = "raw_eeg_features"
                    st.session_state.last_group_cv_predictions = None

                    set_status("Ready")
                    log(
                        f"SVM feature table generated from raw EEG "
                        f"({feature_summary.get('n_rows')} rows, {feature_summary.get('n_features')} features).",
                        now_iso,
                    )

                    save_run(
                        action="generate_svm_features_from_raw_eeg",
                        status="Ready",
                        metrics={
                            "rows": feature_summary.get("n_rows"),
                            "features": feature_summary.get("n_features"),
                            "subjects": feature_summary.get("n_subjects"),
                            "recordings": feature_summary.get("n_recordings"),
                        },
                    )

                    st.success("SVM feature table generated from raw EEG.")
                    st.rerun()

            incomplete_df = st.session_state.raw_manifest_df[
                ~st.session_state.raw_manifest_df["complete_recording"]
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
                <div class="subtle">Upload either a ready CSV or a supported EEG .mat file</div>
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

        svm_source = st.radio(
            "SVM source",
            ["CSV feature table", "EEG .mat file"],
            horizontal=True,
            key="svm_source_choice",
            help="Choose one source type before loading the SVM dataset.",
        )

        uploaded_csv = None
        uploaded_mat = None

        if svm_source == "CSV feature table":
            # for .csv upload
            uploaded_csv = st.file_uploader(
                "Upload CSV dataset",
                type=["csv"],
                accept_multiple_files=False,
                key="csv_uploader",
                help="Ready-made tabular feature table for SVM.",
            )
        else:
            # for .mat upload
            uploaded_mat = st.file_uploader(
                "Upload EEG .mat dataset",
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

        can_load_svm = (
            (svm_source == "CSV feature table" and uploaded_csv is not None)
            or (svm_source == "EEG .mat file" and uploaded_mat is not None)
        )

        if st.button(
            "Load SVM dataset",
            use_container_width=True,
            disabled=not can_load_svm,
            key="load_svm_btn",
        ): # uploaded files are analyzed
            try:
                if svm_source == "CSV feature table" and uploaded_csv is not None:
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

                elif svm_source == "EEG .mat file" and uploaded_mat is not None:
                    st.session_state.mat_file_payload = uploaded_mat.getvalue()
                    st.session_state.mat_file_name = uploaded_mat.name

                    df, summary, err = parse_mat_dataset(
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
                else:
                    st.warning("Choose a source and upload a file before loading the SVM dataset.")

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


def render_preprocess():
    st.markdown(
        """
        <div class="card">
          <div class="card-title">
            <div style="font-weight:800; font-size:1.05rem;">Preprocessing config</div>
            <div class="subtle">Windowing + filtering settings used by Raw EEG CNN and MAT → SVM conversion</div>
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
        "spectrogram_channel_mode": str(SPECTROGRAM_CHANNEL_MODE),
        "spectrogram_nperseg": int(SPECTROGRAM_NPERSEG),
        "spectrogram_overlap_ratio": float(SPECTROGRAM_OVERLAP_RATIO),
    }

    with left:
        c1, c2 = st.columns(2)
        with c1:
            window_sec = st.number_input(
                "Window length (sec)",
                min_value=0.5,
                value=float(default_cfg["window_sec"]),
                step=0.5,
                help="How many seconds of EEG are grouped into one analysis window. Longer windows include more signal context; shorter windows create more samples.",
            )
        with c2:
            step_sec = st.number_input(
                "Step (sec)",
                min_value=0.5,
                value=float(default_cfg["step_sec"]),
                step=0.5,
                help="How far the window moves before creating the next sample. If this is smaller than the window length, windows overlap.",
            )

        c3, c4 = st.columns(2)
        with c3:
            max_windows = st.number_input(
                "Max windows / recording",
                min_value=1,
                value=int(default_cfg["max_windows_per_recording"]),
                step=1,
                help="Maximum number of windows kept from each recording. This limits runtime and keeps long recordings from dominating the dataset.",
            )
        with c4:
            notch_freq = st.number_input(
                "Notch (Hz)",
                min_value=0.0,
                value=float(default_cfg["notch_freq"]),
                step=0.5,
                help="Frequency removed by the notch filter, usually power-line noise such as 50 Hz in Europe or 60 Hz in the US.",
            )

        use_notch = st.checkbox(
            "Use notch filter",
            value=bool(default_cfg["use_notch"]),
            help="Removes a very narrow frequency band, commonly used to reduce electrical mains noise.",
        )
        use_bandpass = st.checkbox(
            "Use band-pass filter",
            value=bool(default_cfg["use_bandpass"]),
            help="Keeps only frequencies between the low and high cutoffs, removing very slow drift and very high-frequency noise.",
        )

        if use_bandpass:
            c5, c6 = st.columns(2)
            with c5:
                bandpass_low = st.number_input(
                    "Band-pass low (Hz)",
                    min_value=0.0,
                    value=float(default_cfg["bandpass_low"]),
                    step=0.1,
                    help="Lower frequency cutoff for the band-pass filter. Frequencies below this value are reduced.",
                )
            with c6:
                bandpass_high = st.number_input(
                    "Band-pass high (Hz)",
                    min_value=0.1,
                    value=float(default_cfg["bandpass_high"]),
                    step=0.5,
                    help="Upper frequency cutoff for the band-pass filter. Frequencies above this value are reduced.",
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
                step=16,
                help="Output image height for each spectrogram used by the CNN. Larger images preserve more detail but use more memory.",
            )
        with c8:
            visual_width = st.number_input(
                "Spectrogram width",
                min_value=32,
                max_value=256,
                value=int(default_cfg.get("visual_width", SPECTROGRAM_WIDTH)),
                step=16,
                help="Output image width for each spectrogram used by the CNN. Larger images preserve more time detail but train more slowly.",
            )

        channel_options = ["mean", "regions", "all"]
        current_channel_mode = str(default_cfg.get("spectrogram_channel_mode", SPECTROGRAM_CHANNEL_MODE)).lower()
        if current_channel_mode not in channel_options:
            current_channel_mode = "regions"

        c9, c10 = st.columns(2)
        with c9:
            spectrogram_channel_mode = st.selectbox(
                "Spectrogram channel mode",
                options=channel_options,
                index=channel_options.index(current_channel_mode),
                help="How EEG channels are combined into CNN image channels. regions groups electrodes by scalp area; all keeps each channel; mean averages all channels.",
            )
        with c10:
            spectrogram_nperseg = st.number_input(
                "Spectrogram nperseg",
                min_value=32,
                max_value=512,
                value=int(default_cfg.get("spectrogram_nperseg", SPECTROGRAM_NPERSEG)),
                step=32,
                help="Number of signal samples used for each short-time frequency calculation. Higher values improve frequency resolution but reduce time resolution.",
            )

        spectrogram_overlap_ratio = st.slider(
            "Spectrogram overlap ratio",
            min_value=0.0,
            max_value=0.9,
            value=float(default_cfg.get("spectrogram_overlap_ratio", SPECTROGRAM_OVERLAP_RATIO)),
            step=0.05,
            help="How much neighboring spectrogram segments overlap. More overlap makes smoother spectrograms but increases computation.",
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
                "spectrogram_channel_mode": str(spectrogram_channel_mode),
                "spectrogram_nperseg": int(spectrogram_nperseg),
                "spectrogram_overlap_ratio": float(spectrogram_overlap_ratio),
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
                f"Spectrogram channel mode: {spectrogram_channel_mode}",
                f"Spectrogram nperseg: {spectrogram_nperseg}",
                f"Spectrogram overlap ratio: {spectrogram_overlap_ratio}",
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
                    df, summary, err = parse_mat_dataset(
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
                <div class="subtle">Used by Run CNN Group CV and by MAT → SVM feature generation</div>
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


def get_last_model_display_name(): # returns the name of the last trained model
    action = st.session_state.last_action

    mapping = {
        "cnn": "CNN",
        "svm": "SVM",
        "svm_group_cv": "SVM Group CV",
        "cnn_group_cv": "CNN Group CV",
        "cnn_raw_eeg": "CNN",
    }

    return mapping.get(action, "Unknown model")


def render_results():
    model_name = get_last_model_display_name()
    metrics = st.session_state.last_metrics or {}
    last_action = st.session_state.last_action

    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">
            <div>
              <div style="font-weight:800; font-size:1.05rem;">Evaluation and model comparison</div>
              <div class="subtle">Metrics, visualizations, diagnostics and traceability</div>
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
        return

    def fmt_value(value, digits=3):
        if value is None:
            return "-"
        try:
            if pd.isna(value):
                return "-"
        except TypeError:
            pass
        if isinstance(value, (int, float)):
            return f"{float(value):.{digits}f}"
        return str(value)

    def label_from_int(value):
        return "PD" if int(value) == 1 else "HC"

    def metric_table(rows):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    def escape_html(value):
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    metric_help = {
        "accuracy": "Share of samples classified correctly.",
        "balanced_accuracy": "Average of sensitivity and specificity. Useful when the HC and PD classes are not perfectly balanced.",
        "sensitivity": "Recall for the PD class: the share of true PD cases correctly detected.",
        "specificity": "Recall for the HC class: the share of true HC cases correctly detected.",
        "subject_acc_mean": "Average subject-level accuracy across CV folds. Subject-level metrics combine all windows from the same subject first.",
        "subject_acc_std": "Variation in subject-level accuracy across CV folds. Lower values mean performance is more stable between folds.",
        "subject_f1_mean": "Average subject-level F1 score across CV folds. F1 balances precision and recall for the PD class.",
        "subject_f1_std": "Variation in subject-level F1 across CV folds. Lower values mean the F1 score is more stable between folds.",
        "subject_auc_mean": "Average subject-level ROC AUC across CV folds. Higher means better separation between HC and PD subjects.",
        "subject_auc_std": "Variation in subject-level ROC AUC across CV folds. Lower values mean AUC is more stable between folds.",
        "subject_accuracy": "Subject-level accuracy for the latest run. Windows from the same subject are combined before scoring.",
        "subject_f1": "Subject-level F1 score for the latest run. F1 balances precision and recall for the PD class.",
        "subject_auc": "Subject-level ROC AUC for the latest run. Higher means better separation between HC and PD subjects.",
        "subject_balanced_accuracy": "Subject-level balanced accuracy for the latest run.",
        "subject_sensitivity": "Subject-level PD recall: the share of true PD subjects correctly detected.",
        "subject_specificity": "Subject-level HC recall: the share of true HC subjects correctly detected.",
        "subject_balanced_accuracy_mean": "Average subject-level balanced accuracy across CV folds.",
        "subject_sensitivity_mean": "Average subject-level PD recall across CV folds.",
        "subject_specificity_mean": "Average subject-level HC recall across CV folds.",
        "window_acc_mean": "Average window-level accuracy across CV folds. Each EEG window is scored separately.",
        "window_acc_std": "Variation in window-level accuracy across CV folds.",
        "window_f1_mean": "Average window-level F1 score across CV folds.",
        "window_f1_std": "Variation in window-level F1 across CV folds.",
        "window_auc_mean": "Average window-level ROC AUC across CV folds.",
        "window_auc_std": "Variation in window-level ROC AUC across CV folds.",
        "window_balanced_accuracy_mean": "Average window-level balanced accuracy across CV folds.",
        "window_sensitivity_mean": "Average window-level PD recall across CV folds.",
        "window_specificity_mean": "Average window-level HC recall across CV folds.",
        "window_mean_proba_pd_for_true_hc": "Average predicted PD probability for windows that are truly HC. Lower is better.",
        "window_mean_proba_pd_for_true_pd": "Average predicted PD probability for windows that are truly PD. Higher is better.",
        "window_cv_mean_proba_pd_for_true_hc": "Average predicted PD probability for true HC windows across all CV folds. Lower is better.",
        "window_cv_mean_proba_pd_for_true_pd": "Average predicted PD probability for true PD windows across all CV folds. Higher is better.",
        "val_subject_auc": "Subject-level validation AUC used while tuning the CNN threshold.",
        "val_subject_auc_mean": "Average subject-level validation AUC across CNN CV folds. SVM Group CV does not use an inner validation split, so this can be unavailable.",
        "subject_auc_if_flipped": "AUC that would result if PD and HC probabilities were reversed. Useful for detecting inverted labels or predictions.",
        "subject_cv_auc_if_flipped": "Cross-validation AUC if PD and HC probabilities were reversed. Useful for spotting inverted labels or predictions.",
        "f1": "F1 score for the PD class, balancing precision and recall.",
        "auc": "ROC AUC, measuring how well the model separates HC from PD across probability thresholds.",
    }

    def render_metric_row(specs):
        cols = st.columns(len(specs))
        for col, (label, key) in zip(cols, specs):
            with col:
                value = fmt_value(metrics.get(key))
                tooltip = metric_help.get(key, f"{label} for the latest run.")
                st.markdown(
                    f"""
                    <div class="metric-card">
                      <div class="metric-label">
                        <span>{escape_html(label)}</span>
                        <span class="metric-tooltip" tabindex="0" aria-label="{escape_html(tooltip)}">
                          ?
                          <span class="metric-tooltip-text">{escape_html(tooltip)}</span>
                        </span>
                      </div>
                      <div class="metric-value">{escape_html(value)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    def split_summary_rows():
        return [
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

    def render_group_metric_tables():
        metric_table([
            {
                "level": "subject",
                "accuracy mean": fmt_value(metrics.get("subject_acc_mean")),
                "accuracy std": fmt_value(metrics.get("subject_acc_std")),
                "F1 mean": fmt_value(metrics.get("subject_f1_mean")),
                "F1 std": fmt_value(metrics.get("subject_f1_std")),
                "AUC mean": fmt_value(metrics.get("subject_auc_mean")),
                "AUC std": fmt_value(metrics.get("subject_auc_std")),
                "balanced accuracy": fmt_value(metrics.get("subject_balanced_accuracy_mean")),
                "sensitivity": fmt_value(metrics.get("subject_sensitivity_mean")),
                "specificity": fmt_value(metrics.get("subject_specificity_mean")),
            },
            {
                "level": "window",
                "accuracy mean": fmt_value(metrics.get("window_acc_mean")),
                "accuracy std": fmt_value(metrics.get("window_acc_std")),
                "F1 mean": fmt_value(metrics.get("window_f1_mean")),
                "F1 std": fmt_value(metrics.get("window_f1_std")),
                "AUC mean": fmt_value(metrics.get("window_auc_mean")),
                "AUC std": fmt_value(metrics.get("window_auc_std")),
                "balanced accuracy": fmt_value(metrics.get("window_balanced_accuracy_mean")),
                "sensitivity": fmt_value(metrics.get("window_sensitivity_mean")),
                "specificity": fmt_value(metrics.get("window_specificity_mean")),
            },
        ])

    def render_single_metric_tables():
        rows = [{
            "level": "window/sample",
            "accuracy": fmt_value(metrics.get("accuracy")),
            "F1": fmt_value(metrics.get("f1")),
            "AUC": fmt_value(metrics.get("auc")),
            "balanced accuracy": fmt_value(metrics.get("balanced_accuracy")),
            "sensitivity": fmt_value(metrics.get("sensitivity")),
            "specificity": fmt_value(metrics.get("specificity")),
        }]

        if last_action == "cnn":
            rows.append({
                "level": "subject",
                "accuracy": fmt_value(metrics.get("subject_accuracy")),
                "F1": fmt_value(metrics.get("subject_f1")),
                "AUC": fmt_value(metrics.get("subject_auc")),
                "balanced accuracy": fmt_value(metrics.get("subject_balanced_accuracy")),
                "sensitivity": fmt_value(metrics.get("subject_sensitivity")),
                "specificity": fmt_value(metrics.get("subject_specificity")),
            })

        metric_table(rows)

    def render_metadata():
        if last_action in ["cnn", "cnn_group_cv"]:
            rows = [{
                "subjects": metrics.get("n_subjects", metrics.get("subjects", "-")),
                "recordings": metrics.get("n_recordings", "-"),
                "raw channels": metrics.get("n_channels", "-"),
                "CNN input channels": metrics.get("cnn_input_channels", "-"),
                "spectrogram": f"{metrics.get('visual_height', '-')} x {metrics.get('visual_width', '-')}",
                "threshold": fmt_value(metrics.get("threshold", metrics.get("threshold_mean", None))),
            }]
        elif last_action == "svm_group_cv":
            rows = [{
                "subjects": metrics.get("n_subjects", metrics.get("subjects", "-")),
                "features": metrics.get("features", "-"),
                "splits": metrics.get("n_splits", "-"),
                "requested splits": metrics.get("requested_n_splits", "-"),
                "threshold": fmt_value(metrics.get("threshold_mean")),
            }]
        else:
            rows = [{"model": model_name}]

        metric_table(rows)

    def render_visualizations():
        if last_action in ["svm_group_cv", "cnn_group_cv"]:
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

    def render_diagnostics():
        if last_action == "cnn":
            render_metric_row([
                ("Mean PD prob | true HC", "window_mean_proba_pd_for_true_hc"),
                ("Mean PD prob | true PD", "window_mean_proba_pd_for_true_pd"),
                ("Val subject AUC", "val_subject_auc"),
                ("Subject AUC if flipped", "subject_auc_if_flipped"),
            ])
            if metrics.get("subject_looks_inverted", False):
                st.warning(
                    "The model gives higher PD probabilities to HC subjects than to PD subjects. "
                    "Check label detection in the raw EEG manifest and inspect filenames."
                )
        elif last_action in ["svm_group_cv", "cnn_group_cv"]:
            group_cv_name = "CNN Group CV" if last_action == "cnn_group_cv" else "SVM Group CV"
            render_metric_row([
                ("Mean PD prob | true HC", "window_cv_mean_proba_pd_for_true_hc"),
                ("Mean PD prob | true PD", "window_cv_mean_proba_pd_for_true_pd"),
                ("Val subject AUC mean", "val_subject_auc_mean"),
                ("Subject AUC if flipped", "subject_cv_auc_if_flipped"),
            ])
            if metrics.get("subject_cv_looks_inverted", False):
                st.warning(
                    "Across CV folds, the model gives higher PD probabilities to HC subjects than to PD subjects. "
                    "Check label detection in the raw EEG manifest and inspect filenames."
                )

            leakage_detected = metrics.get("subject_leakage_detected", None)
            if leakage_detected is False:
                st.success(f"{group_cv_name} uses subject-level outer folds. No subject leakage was detected.")
            elif leakage_detected is True:
                st.error("Subject leakage detected.")
            else:
                st.info("Subject leakage check not available for this run.")
        else:
            st.info("No extra diagnostics are available for this run.")

    def render_fold_details():
        fold_details = metrics.get("fold_details", [])
        if fold_details:
            st.markdown("**Fold details**")
            st.dataframe(pd.DataFrame(fold_details), use_container_width=True, hide_index=True)
        elif last_action == "cnn":
            st.markdown("**Subject split check**")
            leakage_detected = metrics.get("subject_leakage_detected", None)
            if leakage_detected is False:
                st.success("No subject leakage detected between train, validation and test splits.")
            elif leakage_detected is True:
                st.error("Subject leakage detected.")
            else:
                st.info("Subject leakage check not available for this run.")
            st.dataframe(pd.DataFrame(split_summary_rows()), use_container_width=True, hide_index=True)
        else:
            st.info("Fold details are available after running Group CV.")

    def render_sample_prediction():
        if last_action == "cnn":
            pred_df = st.session_state.raw_cnn_predictions
            if pred_df is None or pred_df.empty:
                st.info("Sample prediction is available after a CNN run.")
                return

            sample_idx = st.number_input(
                "Select CNN test sample index",
                min_value=0,
                max_value=len(pred_df) - 1,
                value=0,
                step=1,
                key="sample_prediction_index_cnn",
            )
            row = pred_df.iloc[int(sample_idx)]
            true_label = label_from_int(row["true_label"])
            pred_label = label_from_int(row["pred_label"])
            confidence = float(max(row["proba_pd"], row["proba_hc"]))
            correct = int(row["true_label"]) == int(row["pred_label"])
            info_df = pd.DataFrame([{
                "recording": row.get("recording", ""),
                "subject_key": row.get("subject_key", ""),
                "window_index": int(row.get("window_index", -1)),
                "start_sample": int(row.get("start_sample", -1)),
                "true_label": true_label,
            }])

        elif last_action == "svm":
            df = st.session_state.dataset_df
            model = st.session_state.last_model
            if df is None or model is None:
                st.info("Sample prediction is available after an SVM run.")
                return

            try:
                label_candidates = [c for c in df.columns if c.lower() in ["label", "class", "y", "target"]]
                if not label_candidates:
                    st.warning("No label column found in dataset.")
                    return

                sample_idx = st.number_input(
                    "Select sample index",
                    min_value=0,
                    max_value=len(df) - 1,
                    value=0,
                    step=1,
                    key="sample_prediction_index_svm",
                )

                X_all, y_all, xy_err = get_xy(df)
                if xy_err:
                    st.warning(xy_err)
                    return

                y_true = y_all.iloc[int(sample_idx)]
                X_sample = X_all.iloc[[int(sample_idx)]]
                pred = model.predict(X_sample.values)[0]
                proba = model.predict_proba(X_sample.values)[0]
                true_label = label_from_int(y_true)
                pred_label = label_from_int(pred)
                confidence = float(max(proba))
                correct = int(pred) == int(y_true)
                info_df = pd.DataFrame([{
                    "row_index": int(df.index[int(sample_idx)]),
                    "true_label": true_label,
                }])
                row = {"proba_pd": float(proba[1]), "proba_hc": float(proba[0])}
            except Exception as e:
                st.error(f"Could not generate sample prediction: {e}")
                return

        elif last_action in ["svm_group_cv", "cnn_group_cv"]:
            pred_df = st.session_state.last_group_cv_predictions
            if pred_df is None or pred_df.empty:
                st.info("Sample prediction is available after running Group CV.")
                return

            sample_idx = st.number_input(
                "Select sample index",
                min_value=0,
                max_value=len(pred_df) - 1,
                value=0,
                step=1,
                key="sample_prediction_index_group_cv",
            )
            row = pred_df.iloc[int(sample_idx)]
            true_label = label_from_int(row["true_label"])
            pred_label = label_from_int(row["pred_label"])
            confidence = float(max(row["proba_pd"], row["proba_hc"]))
            correct = int(row["true_label"]) == int(row["pred_label"])
            info_df = pd.DataFrame([{
                "row_index": int(row.get("row_index", -1)),
                "fold": int(row.get("fold", -1)),
                "subject_id": row.get("subject_id", ""),
                "subject_key": row.get("subject_key", ""),
                "window_start": int(row.get("window_start", -1)),
                "true_label": true_label,
            }])
        else:
            st.info("Train a model first.")
            return

        st.dataframe(info_df, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("True label", true_label, help="Ground-truth class for the selected sample or subject.")
        with c2:
            st.metric("Predicted label", pred_label, help="Class predicted by the model using the selected decision threshold.")
        with c3:
            st.metric("Confidence", f"{confidence:.3f}", help="The larger of the HC and PD predicted probabilities.")

        c4, c5 = st.columns(2)
        with c4:
            st.metric("Probability PD", f"{float(row['proba_pd']):.3f}", help="Model-estimated probability that the selected item belongs to the PD class.")
        with c5:
            st.metric("Probability HC", f"{float(row['proba_hc']):.3f}", help="Model-estimated probability that the selected item belongs to the HC class.")

        if correct:
            st.success("Correct prediction")
        else:
            st.error("Incorrect prediction")

    def render_export():
        runs = load_runs(limit=1)
        if not runs:
            st.info("No run record to export yet.")
            return

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

    if "error" in metrics:
        st.error(metrics["error"])

    summary_specs = (
        [
            ("Subject Accuracy", "subject_acc_mean"),
            ("Subject F1", "subject_f1_mean"),
            ("Subject AUC", "subject_auc_mean"),
        ]
        if last_action in ["svm_group_cv", "cnn_group_cv"]
        else [
            ("Accuracy", "accuracy"),
            ("F1", "f1"),
            ("AUC", "auc"),
        ]
    )
    render_metric_row(summary_specs)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    render_metadata()

    overview_tab, viz_tab, diagnostics_tab, folds_tab, sample_tab, export_tab = st.tabs([
        "Overview",
        "Visualizations",
        "Diagnostics",
        "Fold Details",
        "Sample Prediction",
        "Export",
    ])

    with overview_tab:
        if last_action in ["svm_group_cv", "cnn_group_cv"]:
            render_group_metric_tables()
        else:
            render_single_metric_tables()

    with viz_tab:
        render_visualizations()

    with diagnostics_tab:
        render_diagnostics()

    with folds_tab:
        render_fold_details()

    with sample_tab:
        render_sample_prediction()

    with export_tab:
        render_export()


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
        st.info("Load a raw EEG dataset first from the Import page.")
        if st.button("Go to Import", use_container_width=False):
            st.session_state.page = "Import"
            st.rerun()
        return

    manifest_df = st.session_state.raw_manifest_df.copy() # contains info about uploaded recordinggs
    complete_df = manifest_df[manifest_df["complete_recording"]].copy() # only keeps recordings with a full triplet

    if complete_df.empty:
        st.warning("No complete raw EEG recordings are available for visualization.")
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
        preview_cols = [
            c for c in ["recording", "source_format", "subject_key", "label_guess", "label_source"]
            if c in complete_df.columns
        ]
        preview_df = complete_df[preview_cols].copy()
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

    with right:
        import matplotlib.pyplot as plt
        import numpy as np

        channel_to_idx = {name: idx for idx, name in enumerate(available_channels)} # creates a dict to link each channel name to its numeric indesx
        ordered_indices = [channel_to_idx[name] for name in selected_channels] # converts the selected channel names into indices
        selected_data = data[ordered_indices] # extracts only the selected channel sigs from the eeg data

        view_mode_options = ["Signal", "Spectrogram"]
        if hasattr(st, "segmented_control"):
            selected_view_mode = st.segmented_control(
                "Representation",
                options=view_mode_options,
                default="Signal",
                key="raw_viewer_representation",
            )
        else:
            selected_view_mode = st.radio(
                "Representation",
                options=view_mode_options,
                horizontal=True,
                key="raw_viewer_representation",
            )

        if selected_view_mode == "Signal":
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
        else:
            try:
                from scipy.signal import spectrogram

                cfg = st.session_state.preprocessing_summary or {}
                sfreq = float(summary["sampling_rate"])
                n_samples = int(selected_data.shape[1])
                nperseg = min(int(cfg.get("spectrogram_nperseg", SPECTROGRAM_NPERSEG)), n_samples)
                nperseg = max(8, nperseg)
                noverlap = int(nperseg * float(cfg.get("spectrogram_overlap_ratio", SPECTROGRAM_OVERLAP_RATIO)))
                noverlap = min(max(0, noverlap), nperseg - 1)

                freqs, spec_times, pxx = spectrogram(
                    selected_data,
                    fs=sfreq,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    axis=1,
                    scaling="density",
                    mode="psd",
                )

                if bool(cfg.get("use_bandpass", RAW_USE_BANDPASS)):
                    low = float(cfg.get("bandpass_low", RAW_L_FREQ))
                    high = float(cfg.get("bandpass_high", RAW_H_FREQ))
                    freq_mask = (freqs >= low) & (freqs <= high)
                    if freq_mask.any():
                        freqs = freqs[freq_mask]
                        pxx = pxx[:, freq_mask, :]

                spec_db = 10.0 * np.log10(pxx.astype(np.float32) + 1e-12)
                vmin = float(np.percentile(spec_db, 5))
                vmax = float(np.percentile(spec_db, 95))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                    vmin = float(np.nanmin(spec_db))
                    vmax = float(np.nanmax(spec_db))

                spec_fig, axes = plt.subplots(
                    len(selected_channels),
                    1,
                    figsize=(12, max(3.2, 2.25 * len(selected_channels))),
                    sharex=True,
                    squeeze=False,
                )

                last_mesh = None
                for plot_idx, channel_name in enumerate(selected_channels):
                    ax = axes[plot_idx, 0]
                    last_mesh = ax.pcolormesh(
                        spec_times,
                        freqs,
                        spec_db[plot_idx],
                        shading="auto",
                        vmin=vmin,
                        vmax=vmax,
                    )
                    ax.set_ylabel(f"{channel_name}\nHz")
                    ax.grid(False)

                axes[-1, 0].set_xlabel("Time (s)")
                spec_fig.suptitle(f"Channel spectrograms - {selected_recording}")
                spec_fig.tight_layout()
                if last_mesh is not None:
                    spec_fig.colorbar(last_mesh, ax=axes[:, 0], label="Power (dB)", shrink=0.92)
                st.pyplot(spec_fig, clear_figure=True)

            except Exception as e:
                st.warning(f"Could not generate spectrograms: {e}")

        raw_table = {"time_sec": times}
        for channel_name in selected_channels:
            raw_table[channel_name] = data[channel_to_idx[channel_name]]
        raw_table_df = pd.DataFrame(raw_table)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        with st.expander("Preview signal values", expanded=False):
            st.dataframe(raw_table_df.head(300), use_container_width=True, hide_index=True)
