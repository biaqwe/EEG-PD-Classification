import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

import tempfile
from pathlib import Path

from scripts.make_features_iowa import build_iowa_features_from_mat

from src.actions import (
    run_pipeline,
    run_train_cnn,
    run_train_svm,
    run_train_svm_group_cv,
)
from src.data_utils import dataset_summary, parse_csv
from src.state import log, set_status
from src.storage import load_runs, save_run
from src.ui.components import plot_cm, plot_roc, status_dot
from src.utils import now_iso, safe_float


def render_dashboard():
    df = st.session_state.dataset_df
    n_rows, n_channels = dataset_summary(df)
    ds_ok = df is not None

    left, right = st.columns([1.25, 0.75], gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Overview</div>
              </div>
              <div class="subtle">
                Use the menu to import EEG data, configure preprocessing, run training and inspect results.
              </div>
              <div style="height:10px;"></div>
              <div class="kpis">
                <div class="kpi">
                  <div class="lbl">Dataset</div>
                  <div class="val">""" + (st.session_state.dataset_name or "Not loaded") + """</div>
                  <div class="hint">Import from CSV (label required)</div>
                </div>
                <div class="kpi">
                  <div class="lbl">Rows</div>
                  <div class="val">""" + (str(n_rows) if n_rows is not None else "-") + """</div>
                  <div class="hint">Samples/epochs/records</div>
                </div>
                <div class="kpi">
                  <div class="lbl">Channels/Features</div>
                  <div class="val">""" + (str(n_channels) if n_channels is not None else "-") + """</div>
                  <div class="hint">All columns except label</div>
                </div>
                <div class="kpi">
                    <div class="lbl">Run status</div>
                    <div class="val">""" + st.session_state.run_status + """</div>
                    <div class="hint" style="display:flex; align-items:center; gap:10px; margin-top:10px;">
                        """ + status_dot() + """
                    </div>
                </div>
              </div>
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

        runs = load_runs(limit=8)
        if runs:
            df_runs = pd.DataFrame(runs)
            cols = ["timestamp", "action", "status", "dataset_name"]
            cols = [c for c in cols if c in df_runs.columns]
            st.dataframe(df_runs[cols], use_container_width=True, hide_index=True)
        else:
            st.info("No runs saved yet.")

    with right:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Quick actions</div>
                <div class="subtle">One-click workflow</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Train CNN", use_container_width=True, disabled=not ds_ok):
                run_train_cnn()
        with c2:
            if st.button("Train SVM", use_container_width=True, disabled=not ds_ok):
                run_train_svm()

        if st.button("Run pipeline", use_container_width=True, disabled=not ds_ok):
            run_pipeline()

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
            <div style="font-weight:800; font-size:1.05rem;">Import dataset</div>
            <div class="subtle">CSV with label column (label/class/y/target)</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

    colL, colR = st.columns([1.0, 1.0], gap="large")

    with colL:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        ds_name = st.text_input("Dataset name", value=st.session_state.dataset_name or "")
        st.caption("Tip: label values can be PD/HC or 1/0.")

        if st.button("Load dataset", use_container_width=True, disabled=(uploaded is None)):
            df = parse_csv(uploaded)
            if df is None or df.empty:
                set_status("Error")
                log("Import failed: could not parse CSV.", now_iso)
                st.error("Could not parse CSV.")
            else:
                st.session_state.dataset_df = df
                st.session_state.dataset_name = ds_name.strip() or getattr(uploaded, "name", "dataset.csv")
                set_status("Ready")
                log(f"Dataset loaded: {st.session_state.dataset_name} (shape={df.shape})", now_iso)
                save_run(action="import", status="Ready", metrics={"shape": list(df.shape)})

    with colR:
        df = st.session_state.dataset_df
        if df is None:
            st.info("No dataset loaded.")
        else:
            st.markdown(
                """
                <div class="card">
                  <div class="card-title">
                    <div style="font-weight:800; font-size:1.05rem;">Preview</div>
                    <div class="subtle">First rows + basic validation</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.dataframe(df.head(15), use_container_width=True, hide_index=True)

            cols = list(df.columns)
            label_candidates = [c for c in cols if c.lower() in ["label", "class", "y", "target"]]

            if not label_candidates:
                st.warning("Label column not found. Add label/class/y/target.")
            else:
                ycol = label_candidates[0]
                uniq = df[ycol].dropna().unique()
                st.success(f"Label column: {ycol} | classes: {list(uniq)[:8]}")


def render_preprocess():
    st.markdown(
        """
        <div class="card">
          <div class="card-title">
            <div style="font-weight:800; font-size:1.05rem;">Preprocessing</div>
            <div class="subtle">Upload Iowa .mat and generate the feature dataset directly in the app</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.0, 1.0], gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Iowa raw EEG input</div>
                <div class="subtle">Generate features from the .mat file</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        uploaded_mat = st.file_uploader("Upload IowaData.mat", type=["mat"], key="mat_uploader")

        c1, c2 = st.columns(2)
        with c1:
            fs = st.number_input("Sampling rate (Hz)", min_value=1.0, value=1000.0, step=1.0)
        with c2:
            notch_freq = st.number_input("Notch (Hz)", min_value=0.0, value=50.0, step=0.5)

        c3, c4 = st.columns(2)
        with c3:
            window = st.number_input("Window (samples)", min_value=100, value=2000, step=100)
        with c4:
            step = st.number_input("Step (samples)", min_value=100, value=2000, step=100)

        c5, c6 = st.columns(2)
        with c5:
            max_windows = st.number_input("Max windows / subject", min_value=1, value=30, step=1)
        with c6:
            use_notch = st.checkbox("Use notch filter", value=True)

        use_bandpass = st.checkbox("Use band-pass filter", value=False)

        if use_bandpass:
            c7, c8 = st.columns(2)
            with c7:
                bandpass_low = st.number_input("Band-pass low (Hz)", min_value=0.0, value=0.5, step=0.1)
            with c8:
                bandpass_high = st.number_input("Band-pass high (Hz)", min_value=0.1, value=40.0, step=0.5)
        else:
            bandpass_low = 0.5
            bandpass_high = 40.0

        dataset_name = st.text_input(
            "Generated dataset name",
            value="iowa_preprocessed_from_mat.csv"
        )

        run_btn = st.button(
            "Run preprocessing from .mat",
            use_container_width=True,
            disabled=(uploaded_mat is None)
        )

        if run_btn:
            try:
                set_status("Running")
                log("Preprocessing from uploaded .mat started.", now_iso)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
                    tmp.write(uploaded_mat.getbuffer())
                    tmp_path = Path(tmp.name)

                df, summary = build_iowa_features_from_mat(
                    mat_path=tmp_path,
                    fs=float(fs),
                    window=int(window),
                    step=int(step),
                    max_windows_per_subject=int(max_windows),
                    use_bandpass=bool(use_bandpass),
                    use_notch=bool(use_notch),
                    bandpass_low=float(bandpass_low),
                    bandpass_high=float(bandpass_high),
                    notch_freq=float(notch_freq),
                    verbose=False,
                )

                st.session_state.dataset_df = df
                st.session_state.dataset_name = dataset_name.strip() or uploaded_mat.name.replace(".mat", ".csv")
                st.session_state.preprocessing_summary = summary
                st.session_state.preprocessing_logs = [
                    f"Input file: {uploaded_mat.name}",
                    f"Rows generated: {len(df)}",
                    f"Columns generated: {len(df.columns)}",
                    f"Valid channels: {summary.get('valid_channels')}",
                    f"Bandpass enabled: {summary.get('use_bandpass')}",
                    f"Notch enabled: {summary.get('use_notch')}",
                ]

                save_run(
                    action="preprocessing_from_mat",
                    status="Ready",
                    metrics={
                        "rows": int(len(df)),
                        "cols": int(len(df.columns)),
                        "summary": summary,
                    },
                )

                set_status("Ready")
                log(
                    f"Preprocessing finished. Generated dataset shape={df.shape}.",
                    now_iso
                )
                st.success(f"Preprocessing complete. Generated dataset shape: {df.shape}")

            except Exception as e:
                set_status("Error")
                log(f"Preprocessing failed: {e}", now_iso)
                st.error(f"Preprocessing failed: {e}")

    with right:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Current generated dataset</div>
                <div class="subtle">Stored directly in session for training</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        df = st.session_state.dataset_df
        if df is None:
            st.info("No dataset is currently loaded/generated.")
        else:
            n_rows, n_channels = dataset_summary(df)
            st.markdown(
                f"""
                <div class="card">
                  <div class="small">
                    <b>Name:</b> {st.session_state.dataset_name or "-"}<br/>
                    <b>Rows:</b> {n_rows}<br/>
                    <b>Features/Channels:</b> {n_channels}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download generated CSV",
                data=csv_bytes,
                file_name=st.session_state.dataset_name or "generated_iowa_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Preprocessing summary</div>
                <div class="subtle">Last executed configuration</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        if st.session_state.preprocessing_summary is None:
            st.info("No preprocessing summary available yet.")
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
                - Upload raw Iowa <code>.mat</code> EEG data<br/>
                - Apply selected preprocessing options<br/>
                - Segment into fixed windows<br/>
                - Extract time-domain features per channel<br/>
                - Store the generated dataset directly in the app<br/>
                - Use the generated dataset immediately with SVM / SVM Group CV
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.preprocessing_logs:
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            logs_text = "\n".join(st.session_state.preprocessing_logs)
            st.markdown(
                f"<div class='logbox'>{logs_text.replace('<','&lt;').replace('>','&gt;')}</div>",
                unsafe_allow_html=True
            )

def render_results():
    st.markdown(
        """
        <div class="card">
          <div class="card-title">
            <div style="font-weight:800; font-size:1.05rem;">Evaluation and model comparison</div>
            <div class="subtle">Metrics, confusion matrix, ROC/AUC, export</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

    colL, colR = st.columns([1.05, 0.95], gap="large")

    with colL:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Metrics</div>
                <div class="subtle">Last run</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        metrics = st.session_state.last_metrics or {}
        if not metrics:
            st.info("No metrics yet. Train a model first.")
        else:
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

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

                st.markdown("**Window-level mean performance (Group CV)**")
                mcols2 = st.columns(3)
                with mcols2[0]:
                    st.metric(
                        "Window Accuracy",
                        "-" if metrics.get("window_acc_mean") is None else f"{metrics['window_acc_mean']:.3f}"
                    )
                with mcols2[1]:
                    st.metric(
                        "Window F1",
                        "-" if metrics.get("window_f1_mean") is None else f"{metrics['window_f1_mean']:.3f}"
                    )
                with mcols2[2]:
                    st.metric(
                        "Window AUC",
                        "-" if metrics.get("window_auc_mean") is None else f"{metrics['window_auc_mean']:.3f}"
                    )

            else:
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

            if "error" in metrics:
                st.error(metrics["error"])

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

        if st.session_state.last_action == "svm_group_cv":
            if st.session_state.last_cm_subject is not None:
                st.markdown("**Subject-level confusion matrix**")
                plot_cm(st.session_state.last_cm_subject, title="Subject-level Confusion Matrix")
            else:
                st.info("Subject-level confusion matrix not available yet.")

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            if st.session_state.last_cm_window is not None:
                st.markdown("**Window-level confusion matrix**")
                plot_cm(st.session_state.last_cm_window, title="Window-level Confusion Matrix")
            else:
                st.info("Window-level confusion matrix not available yet.")
        else:
            if st.session_state.last_cm is not None:
                plot_cm(st.session_state.last_cm)
            else:
                st.info("Confusion matrix not available yet.")

            if st.session_state.last_roc is not None:
                plot_roc(st.session_state.last_roc)

    with colR:
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

        runs = load_runs(limit=1)
        if not runs:
            st.info("No run record to export yet.")
        else:
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

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Model actions</div>
                <div class="subtle">SVM requires label column</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Train SVM", use_container_width=True, disabled=(st.session_state.dataset_df is None)):
                run_train_svm()
        with c2:
            if st.button("Train CNN", use_container_width=True, disabled=(st.session_state.dataset_df is None)):
                run_train_cnn()

        if st.button("Run SVM Group CV", use_container_width=True, disabled=(st.session_state.dataset_df is None)):
            run_train_svm_group_cv()