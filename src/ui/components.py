import numpy as np
import streamlit as st

from src.data_utils import dataset_summary

try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False


def escape_html(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def spacer(size: str = "md"):
    allowed = {"xs", "sm", "md", "lg"}
    safe_size = size if size in allowed else "md"
    st.markdown(f"<div class='spacer-{safe_size}'></div>", unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str | None = None, aside: str | None = None):
    subtitle_html = f'<div class="section-subtitle">{escape_html(subtitle)}</div>' if subtitle else ""
    aside_html = f'<div class="section-aside">{aside}</div>' if aside else ""
    st.markdown(
        f"""
        <div class="section-heading">
          <div>
            <div class="section-title">{escape_html(title)}</div>
            {subtitle_html}
          </div>
          {aside_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, message: str, tone: str = "info"):
    tone_cls = {
        "info": "empty-info",
        "warn": "empty-warn",
        "error": "empty-error",
        "success": "empty-success",
    }.get(tone, "empty-info")

    st.markdown(
        f"""
        <div class="empty-state {tone_cls}">
          <div class="empty-title">{escape_html(title)}</div>
          <div class="empty-copy">{escape_html(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        '<h2 class="app-title">EEG-Based Parkinson&#39;s Classification</h2>'
        '<div class="app-subtitle">'
        "Signal analysis and model evaluation for Parkinson's vs healthy control EEG data"
        "</div>"
        '<div class="pill-row">'
        f'<span class="pill">Dataset: <b>{escape_html(ds_name)}</b></span>'
        f'<span class="pill">Rows/Recordings: <b>{escape_html(n_rows if n_rows is not None else "-")}</b></span>'
        f'<span class="pill">Channels/Features: <b>{escape_html(n_channels if n_channels is not None else "-")}</b></span>'
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
    fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor="#0d121a")
    ax.set_facecolor("#111923")
    image = ax.imshow(arr, interpolation="nearest", cmap="mako" if "mako" in plt.colormaps() else "viridis")
    ax.set_title(title, color="#edf4fb", pad=14, fontweight="bold")
    ax.set_xlabel("Predicted", color="#b9c7d6")
    ax.set_ylabel("True", color="#b9c7d6")
    ax.tick_params(colors="#b9c7d6")
    for spine in ax.spines.values():
        spine.set_color("#2a3748")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", color="#f8fbff", fontweight="bold")

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#b9c7d6")
    plt.setp(cbar.ax.get_yticklabels(), color="#b9c7d6")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def plot_roc(roc):
    if not MPL_OK or roc is None:
        return

    fpr = np.array(roc["fpr"], dtype=float)
    tpr = np.array(roc["tpr"], dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor="#0d121a")
    ax.set_facecolor("#111923")
    ax.plot(fpr, tpr, color="#4fd1c5", linewidth=2.4)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7f8ea3", linewidth=1.2)
    ax.set_xlabel("False Positive Rate", color="#b9c7d6")
    ax.set_ylabel("True Positive Rate", color="#b9c7d6")
    ax.set_title("ROC Curve", color="#edf4fb", pad=14, fontweight="bold")
    ax.grid(True, color="#263242", alpha=0.6, linewidth=0.8)
    ax.tick_params(colors="#b9c7d6")
    for spine in ax.spines.values():
        spine.set_color("#2a3748")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
