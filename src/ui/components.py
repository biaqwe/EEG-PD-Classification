import numpy as np
import streamlit as st

from src.data_utils import dataset_summary

try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False


def badge(text: str, tone: str):
    tone_map = {
        "idle": "badge badge-idle",
        "ok": "badge badge-ok",
        "warn": "badge badge-warn",
        "err": "badge badge-err",
        "run": "badge badge-run",
    }
    cls = tone_map.get(tone, "badge")
    return f"<span class='{cls}'>{text}</span>"


def status_badge():
    s = st.session_state.run_status
    if s.lower() in ["idle"]:
        return badge("Idle", "idle")
    if s.lower() in ["running", "processing"]:
        return badge("Running", "run")
    if s.lower() in ["ready", "ok", "done", "completed"]:
        return badge("Ready", "ok")
    if s.lower() in ["warning", "partial"]:
        return badge("Warning", "warn")
    if s.lower() in ["error", "failed"]:
        return badge("Error", "err")
    return badge(s, "warn")


def status_dot():
    s = st.session_state.run_status.lower()

    if s in ["idle"]:
        cls = "dot-idle"
    elif s in ["running", "processing"]:
        cls = "dot-run"
    elif s in ["ready", "ok", "done", "completed"]:
        cls = "dot-ok"
    elif s in ["warning", "partial"]:
        cls = "dot-warn"
    elif s in ["error", "failed"]:
        cls = "dot-err"
    else:
        cls = "dot-warn"

    return f"<span class='status-dot {cls}'></span>"


def render_topbar():
    df = st.session_state.dataset_df
    ds_name = st.session_state.dataset_name or st.session_state.raw_dataset_name or "No dataset"
    page = st.session_state.page

    if df is not None:
        n_rows, n_channels = dataset_summary(df)
    elif st.session_state.raw_dataset_summary is not None:
        summary = st.session_state.raw_dataset_summary
        n_rows = summary.get("n_recordings", "-")
        n_channels = summary.get("n_channels", "-")
    else:
        n_rows, n_channels = None, None

    help_html = ""
    if page == "Import":
        help_html = (
            '<div class="help-card topbar-help">'
            '<div class="help-title">How to use</div>'
            '<div class="small">'
            '- Upload raw EEG recordings when you want to run <b>CNN Group CV</b>.<br/>'
            '- Upload a ready CSV when you already have tabular <b>SVM</b> features.<br/>'
            '- Upload an EEG .mat file when you want the app to generate <b>SVM</b> features using the saved preprocessing config.'
            '</div>'
            '</div>'
        )
    elif page == "Preprocess":
        help_html = (
            '<div class="help-card topbar-help">'
            '<div class="help-title">What this does</div>'
            '<div class="small">'
            '- Controls how EEG is filtered, windowed and converted before modelling.<br/>'
            '- Used when generating spectrograms for <b>CNN Group CV</b>.<br/>'
            '- Also used when converting EEG .mat files into tabular <b>SVM</b> features.'
            '</div>'
            '</div>'
        )

    main_html = (
        "<div>"
        '<h2 style="margin:0; padding:0;">EEG-Based Parkinson&#39;s Classification</h2>'
        '<div class="subtle" style="margin-top:2px;">'
        "Signal analysis and model evaluation for Parkinson's vs healthy control EEG data"
        "</div>"
        '<div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">'
        f'<span class="pill">Dataset: <b style="color:var(--txt)">{ds_name}</b></span>'
        f'<span class="pill">Rows/Recordings: <b style="color:var(--txt)">{n_rows if n_rows is not None else "-"}</b></span>'
        f'<span class="pill">Channels/Features: <b style="color:var(--txt)">{n_channels if n_channels is not None else "-"}</b></span>'
        f'<span class="pill">Status: {status_badge()}</span>'
        "</div>"
        "</div>"
    )
    content_html = (
        f'<div class="topbar-grid">{main_html}<div class="topbar-help-wrap">{help_html}</div></div>'
        if help_html
        else main_html
    )

    st.markdown(
        f'<div class="topbar">{content_html}</div>',
        unsafe_allow_html=True,
    )


def plot_cm(cm, title="Confusion Matrix"):
    if not MPL_OK:
        st.write(cm)
        return

    arr = np.array(cm, dtype=float)
    fig = plt.figure(figsize=(5.2, 4.0))
    plt.imshow(arr, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, int(arr[i, j]), ha="center", va="center")

    st.pyplot(fig, clear_figure=True)


def plot_roc(roc):
    if not MPL_OK or roc is None:
        return

    fpr = np.array(roc["fpr"], dtype=float)
    tpr = np.array(roc["tpr"], dtype=float)

    fig = plt.figure(figsize=(5.2, 4.0))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    st.pyplot(fig, clear_figure=True)
