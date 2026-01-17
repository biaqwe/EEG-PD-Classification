import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EEG Classification (PD vs HC)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ascunde header/toolbar */
header[data-testid="stHeader"] { display: none; }
div[data-testid="stToolbar"] { display: none; }

/* optional: elimina spatiul de sus din containerul principal */
div[data-testid="stAppViewContainer"] { padding-top: 0rem; }

/* SIDEBAR container */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(11,16,32,0.95), rgba(15,23,48,0.95)) !important;
  border-right: 1px solid rgba(255,255,255,0.12);
  box-shadow: 8px 0 30px rgba(0,0,0,0.35);
}

/* AICI e spatiul real in Streamlit 1.53.0 */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
  padding-top: -1.2rem !important;   /* urca meniul */
  padding-bottom: 0.6rem !important;
}

/* titluri sidebar */
section[data-testid="stSidebar"] .sidebar-title{
  color: rgba(255,255,255,0.55);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin: 0.2rem 0 10px 4px !important;
}

/* butoane */
section[data-testid="stSidebar"] button{
  width: 100%;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.04) !important;
  color: rgba(255,255,255,0.92) !important;
  padding: 0.75rem 0.9rem !important;
  font-size: 0.95rem !important;
  transition: all 0.15s ease-in-out;
}

section[data-testid="stSidebar"] button:hover{
  background: rgba(106,166,255,0.12) !important;
  border-color: rgba(106,166,255,0.45) !important;
  transform: translateY(-1px);
}

.navbtn-active > button{
  background: linear-gradient(180deg, rgba(106,166,255,0.18), rgba(106,166,255,0.08)) !important;
  border-color: rgba(106,166,255,0.65) !important;
  box-shadow: 0 0 0 1px rgba(106,166,255,0.25);
}

/* spacing pe butoane */
section[data-testid="stSidebar"] .stButton{ margin-bottom: 8px; }

/* text mic */
section[data-testid="stSidebar"] .small{
  color: rgba(255,255,255,0.6);
  font-size: 0.85rem;
  line-height: 1.4;
}

/* transparent pe blocuri interne */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
  background: transparent !important;
  padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)


APP_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
RUNS_DIR = APP_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False


def _now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default):
    try:
        return int(x)
    except Exception:
        return default


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


BASE_CSS = """
<style>
  :root{
    --bg0:#0b1020;
    --bg1:#0f1730;
    --card:#101a33;
    --card2:#0f1a2f;
    --stroke: rgba(255,255,255,.10);
    --stroke2: rgba(255,255,255,.16);
    --txt: rgba(255,255,255,.90);
    --muted: rgba(255,255,255,.70);
    --muted2: rgba(255,255,255,.55);
    --blue:#6aa6ff;
    --cyan:#4fe3d5;
    --lime:#bafc5a;
    --amber:#ffc857;
    --red:#ff4d6d;
    --purple:#a78bfa;
  }

  html, body, [data-testid="stAppViewContainer"]{
    background: radial-gradient(1200px 700px at 15% 10%, rgba(106,166,255,.14), transparent 60%),
                radial-gradient(900px 650px at 85% 20%, rgba(79,227,213,.12), transparent 55%),
                radial-gradient(900px 650px at 55% 95%, rgba(167,139,250,.12), transparent 55%),
                linear-gradient(180deg, var(--bg0), var(--bg1));
    color: var(--txt);
  }

  .block-container{
    padding-top: 1.0rem;
    padding-bottom: 1.8rem;
    max-width: 1350px;
  }

  h1,h2,h3{
    letter-spacing: -0.4px;
  }

  .topbar{
    border: 1px solid var(--stroke);
    background: linear-gradient(180deg, rgba(16,26,51,.85), rgba(16,26,51,.55));
    border-radius: 18px;
    padding: 14px 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,.25);
  }

  .subtle{
    color: var(--muted);
    font-size: 0.95rem;
  }

  .pill{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:6px 10px;
    border-radius: 999px;
    border: 1px solid var(--stroke);
    background: rgba(255,255,255,.03);
    color: var(--muted);
    font-size: 0.9rem;
  }

  .grid{
    display:grid;
    grid-template-columns: 1.2fr 0.8fr;
    gap: 14px;
  }
  @media (max-width: 1100px){
    .grid{ grid-template-columns: 1fr; }
  }

  .card{
    border: 1px solid var(--stroke);
    background: linear-gradient(180deg, rgba(16,26,51,.75), rgba(16,26,51,.45));
    border-radius: 18px;
    padding: 14px 16px;
    box-shadow: 0 10px 28px rgba(0,0,0,.22);
  }

  .card-title{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap: 12px;
    margin-bottom: 6px;
  }

  .kpis{
    display:grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
  }
  @media (max-width: 1100px){
    .kpis{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }

  .kpi{
    border: 1px solid var(--stroke);
    background: rgba(255,255,255,.03);
    border-radius: 16px;
    padding: 12px 12px;
  }
  .kpi .lbl{ color: var(--muted2); font-size: 0.85rem; }
  .kpi .val{ font-size: 1.25rem; font-weight: 700; margin-top: 4px; }
  .kpi .hint{ color: var(--muted); font-size: 0.88rem; margin-top: 6px; }

  .btnrow{
    display:flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 10px;
  }

  .badge{
    display:inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    border: 1px solid var(--stroke2);
    background: rgba(255,255,255,.04);
    font-size: .85rem;
    color: var(--muted);
  }
  .badge-idle{ border-color: rgba(255,200,87,.38); background: rgba(255,200,87,.09); color: rgba(255,230,180,.95); }
  .badge-ok{ border-color: rgba(186,252,90,.32); background: rgba(186,252,90,.10); color: rgba(233,255,205,.95); }
  .badge-warn{ border-color: rgba(255,200,87,.32); background: rgba(255,200,87,.10); color: rgba(255,230,180,.95); }
  .badge-err{ border-color: rgba(255,77,109,.34); background: rgba(255,77,109,.10); color: rgba(255,205,215,.95); }
  .badge-run{ border-color: rgba(79,227,213,.32); background: rgba(79,227,213,.10); color: rgba(205,255,248,.95); }

  .sidebar-title{
    font-size: 0.9rem;
    color: var(--muted2);
    text-transform: uppercase;
    letter-spacing: .12em;
    margin: 6px 0 10px 0;
  }

  .navbtn > button{
    width: 100%;
    border-radius: 14px !important;
    border: 1px solid var(--stroke) !important;
    background: rgba(255,255,255,.03) !important;
    color: var(--txt) !important;
    padding: 0.6rem 0.75rem !important;
    transition: transform .05s ease-in-out, border-color .12s ease-in-out;
  }
  .navbtn > button:hover{
    border-color: rgba(106,166,255,.35) !important;
  }
  .navbtn-active > button{
    border-color: rgba(106,166,255,.55) !important;
    background: rgba(106,166,255,.10) !important;
  }

  .small{
    color: var(--muted2);
    font-size: .88rem;
  }

  .logbox{
    border: 1px dashed var(--stroke2);
    background: rgba(255,255,255,.02);
    border-radius: 14px;
    padding: 10px 12px;
    max-height: 240px;
    overflow: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.86rem;
    color: rgba(255,255,255,.82);
  }

  .status-dot{
    width: 12px;
    height: 12px;
    border-radius: 999px;
    display: inline-block;
    border: 1px solid var(--stroke2);
    box-shadow: 0 0 0 3px rgba(255,255,255,.03);
  }

  .dot-idle{ background: rgba(255,200,87,.95); box-shadow: 0 0 12px rgba(255,200,87,.22), 0 0 0 3px rgba(255,200,87,.06); }
  .dot-run{  background: rgba(79,227,213,.95); box-shadow: 0 0 12px rgba(79,227,213,.22), 0 0 0 3px rgba(79,227,213,.06); }
  .dot-ok{   background: rgba(186,252,90,.95); box-shadow: 0 0 12px rgba(186,252,90,.20), 0 0 0 3px rgba(186,252,90,.06); }
  .dot-warn{ background: rgba(255,200,87,.95); box-shadow: 0 0 12px rgba(255,200,87,.18), 0 0 0 3px rgba(255,200,87,.06); }
  .dot-err{  background: rgba(255,77,109,.95); box-shadow: 0 0 12px rgba(255,77,109,.22), 0 0 0 3px rgba(255,77,109,.06); }

</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)


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


def ss_init():
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "dataset_df" not in st.session_state:
        st.session_state.dataset_df = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None
    if "preproc" not in st.session_state:
        st.session_state.preproc = PreprocConfig()
    if "run_status" not in st.session_state:
        st.session_state.run_status = "Idle"
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = {}
    if "last_cm" not in st.session_state:
        st.session_state.last_cm = None
    if "last_roc" not in st.session_state:
        st.session_state.last_roc = None
    if "last_action" not in st.session_state:
        st.session_state.last_action = None


def log(msg: str):
    st.session_state.logs.append(f"[{_now_iso()}] {msg}")


def set_status(new_status: str):
    st.session_state.run_status = new_status


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


def dataset_summary(df: Optional[pd.DataFrame]) -> Tuple[Optional[int], Optional[int]]:
    if df is None:
        return None, None
    cols = list(df.columns)
    label_cols = [c for c in cols if c.lower() in ["label", "class", "y", "target"]]
    feature_cols = [c for c in cols if c not in label_cols]
    n_rows = len(df)
    n_channels = len(feature_cols)
    return n_rows, n_channels


def save_run(action: str, status: str, metrics: dict):
    df = st.session_state.dataset_df
    n_rows, n_channels = dataset_summary(df)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    rec = RunRecord(
        run_id=run_id,
        timestamp=_now_iso(),
        dataset_name=st.session_state.dataset_name,
        n_rows=n_rows,
        n_channels=n_channels,
        preproc=asdict(st.session_state.preproc),
        action=action,
        status=status,
        metrics=metrics or {},
    )
    path = RUNS_DIR / f"{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(rec), f, ensure_ascii=False, indent=2)
    log(f"Saved run record: {path.name}")


def load_runs(limit: int = 30):
    items = sorted(RUNS_DIR.glob("*.json"), reverse=True)
    out = []
    for p in items[:limit]:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


def parse_csv(uploaded) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(uploaded)
        return df
    except Exception:
        try:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, sep=";")
            return df
        except Exception:
            return None


def get_xy(df: pd.DataFrame):
    cols = list(df.columns)
    label_candidates = [c for c in cols if c.lower() in ["label", "class", "y", "target"]]
    if not label_candidates:
        return None, None, "Dataset CSV must contain a label column: label/class/y/target."
    ycol = label_candidates[0]
    X = df.drop(columns=[ycol]).copy()
    y = df[ycol].copy()
    if y.dtype == object:
        y = y.astype(str).str.strip()
        y = y.map({"pd": 1, "hc": 0, "1": 1, "0": 0}).fillna(y)
    try:
        y = y.astype(int)
    except Exception:
        uniq = sorted(pd.unique(y))
        mapping = {v: i for i, v in enumerate(uniq)}
        y = y.map(mapping).astype(int)
    return X, y, None


def train_svm(df: pd.DataFrame):
    if not SKLEARN_OK:
        return None, None, None, None, "scikit-learn not available in this environment."
    X, y, err = get_xy(df)
    if err:
        return None, None, None, None, err
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = y.loc[X.index]
    if len(X) < 10 or len(np.unique(y)) < 2:
        return None, None, None, None, "Not enough data or only one class present."
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.25, random_state=42, stratify=y.values
    )
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
        ]
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred))
    cm = confusion_matrix(y_test, pred).tolist()

    auc = None
    roc = None
    try:
        auc = float(roc_auc_score(y_test, proba))
        fpr, tpr, thr = roc_curve(y_test, proba)
        roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": thr.tolist()}
    except Exception:
        pass

    metrics = {"accuracy": acc, "f1": f1}
    if auc is not None:
        metrics["auc"] = auc
    return metrics, cm, roc, clf, None


def fake_cnn_result():
    rng = np.random.default_rng(42)
    acc = float(np.clip(rng.normal(0.82, 0.05), 0.60, 0.95))
    f1 = float(np.clip(rng.normal(0.80, 0.06), 0.55, 0.95))
    auc = float(np.clip(rng.normal(0.86, 0.05), 0.60, 0.98))
    cm = [[int(rng.integers(18, 30)), int(rng.integers(2, 10))],
          [int(rng.integers(3, 12)), int(rng.integers(16, 30))]]
    roc = {"fpr": [0.0, 0.08, 0.18, 0.35, 1.0], "tpr": [0.0, 0.55, 0.72, 0.87, 1.0], "thr": [1.2, 0.75, 0.52, 0.28, 0.0]}
    return {"accuracy": acc, "f1": f1, "auc": auc}, cm, roc


def render_topbar():
    df = st.session_state.dataset_df
    n_rows, n_channels = dataset_summary(df)
    ds_name = st.session_state.dataset_name or "No dataset"

    st.markdown(
        f"""
        <div class="topbar">
          <div>
            <h2 style="margin:0; padding:0;">EEG Classification (PD vs HC)</h2>
            <div class="subtle" style="margin-top:2px;">
              Intelligent system for EEG signal analysis and classification (PD vs Healthy Controls)
            </div>
            <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">
              <span class="pill">Dataset: <b style="color:var(--txt)">{ds_name}</b></span>
              <span class="pill">Rows: <b style="color:var(--txt)">{n_rows if n_rows is not None else "-"}</b></span>
              <span class="pill">Channels/Features: <b style="color:var(--txt)">{n_channels if n_channels is not None else "-"}</b></span>
              <span class="pill">Status: {status_badge()}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def sidebar_nav():
    st.sidebar.markdown("<div class='sidebar-title'>Menu</div>", unsafe_allow_html=True)

    def nav_button(label: str):
        active = (st.session_state.page == label)
        cls = "navbtn navbtn-active" if active else "navbtn"
        with st.sidebar.container():
            st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)
            clicked = st.button(label, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        if clicked:
            st.session_state.page = label

    nav_button("Dashboard")
    nav_button("Import")
    nav_button("Preprocess")
    nav_button("Results")

    st.sidebar.markdown("<div class='sidebar-title'>Run</div>", unsafe_allow_html=True)

    colA, colB = st.sidebar.columns(2)
    with colA:
        if st.button("Run pipeline", use_container_width=True):
            run_pipeline()
    with colB:
        if st.button("Clear logs", use_container_width=True):
            st.session_state.logs = []
            log("Logs cleared.")

    st.sidebar.markdown("<div class='sidebar-title'>Models</div>", unsafe_allow_html=True)

    colC, colD = st.sidebar.columns(2)
    with colC:
        if st.button("Train SVM", use_container_width=True):
            run_train_svm()
    with colD:
        if st.button("Train CNN", use_container_width=True):
            run_train_cnn()


def run_pipeline():
    if st.session_state.dataset_df is None:
        set_status("Error")
        log("Cannot run pipeline: dataset not loaded.")
        save_run(action="pipeline", status="Error", metrics={"error": "dataset not loaded"})
        return
    set_status("Running")
    st.session_state.last_action = "pipeline"
    log("Pipeline started.")
    time.sleep(0.25)
    log(f"Preproc config: {asdict(st.session_state.preproc)}")
    time.sleep(0.25)
    set_status("Ready")
    log("Pipeline finished.")
    save_run(action="pipeline", status="Ready", metrics={"note": "preprocessing simulated"})


def run_train_svm():
    if st.session_state.dataset_df is None:
        set_status("Error")
        log("Cannot train SVM: dataset not loaded.")
        save_run(action="svm", status="Error", metrics={"error": "dataset not loaded"})
        return
    set_status("Running")
    st.session_state.last_action = "svm"
    log("Training SVM started.")
    metrics, cm, roc, model, err = train_svm(st.session_state.dataset_df)
    if err:
        set_status("Error")
        log(f"SVM failed: {err}")
        st.session_state.last_metrics = {"error": err}
        st.session_state.last_cm = None
        st.session_state.last_roc = None
        save_run(action="svm", status="Error", metrics={"error": err})
        return
    st.session_state.last_metrics = metrics
    st.session_state.last_cm = cm
    st.session_state.last_roc = roc
    set_status("Ready")
    log(f"SVM done. Metrics: {metrics}")
    save_run(action="svm", status="Ready", metrics=metrics)


def run_train_cnn():
    if st.session_state.dataset_df is None:
        set_status("Error")
        log("Cannot train CNN: dataset not loaded.")
        save_run(action="cnn", status="Error", metrics={"error": "dataset not loaded"})
        return
    set_status("Running")
    st.session_state.last_action = "cnn"
    log("Training CNN started (demo mode).")
    time.sleep(0.35)
    metrics, cm, roc = fake_cnn_result()
    st.session_state.last_metrics = metrics
    st.session_state.last_cm = cm
    st.session_state.last_roc = roc
    set_status("Ready")
    log(f"CNN done. Metrics: {metrics}")
    save_run(action="cnn", status="Ready", metrics=metrics)


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
                run_pipeline()
        with c2:
            if st.button("Train SVM", use_container_width=True, disabled=not ds_ok):
                run_train_svm()

        if st.button("Run pipeline", use_container_width=True, disabled=not ds_ok):
            run_train_cnn()

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
        st.markdown(f"<div class='logbox'>{logs_text.replace('<','&lt;').replace('>','&gt;')}</div>", unsafe_allow_html=True)


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
                log("Import failed: could not parse CSV.")
                st.error("Could not parse CSV.")
            else:
                st.session_state.dataset_df = df
                st.session_state.dataset_name = ds_name.strip() or getattr(uploaded, "name", "dataset.csv")
                set_status("Ready")
                log(f"Dataset loaded: {st.session_state.dataset_name} (shape={df.shape})")
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
            <div style="font-weight:800; font-size:1.05rem;">Preprocessing configuration</div>
            <div class="subtle">Band-pass, notch, epoching, normalization</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

    cfg = st.session_state.preproc
    col1, col2 = st.columns([0.95, 1.05], gap="large")

    with col1:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Parameters</div>
                <div class="subtle">Defaults match the prototype</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            bp_low = st.number_input(
                "Band-pass low (Hz)",
                min_value=0.0,
                max_value=200.0,
                value=float(cfg.bandpass_low),
                step=0.1,
                key="bp_low",
            )
        with r1c2:
            bp_high = st.number_input(
                "Band-pass high (Hz)",
                min_value=0.0,
                max_value=200.0,
                value=float(cfg.bandpass_high),
                step=0.5,
                key="bp_high",
            )

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            notch = st.number_input(
                "Notch (Hz)",
                min_value=0.0,
                max_value=200.0,
                value=float(cfg.notch),
                step=0.5,
                key="notch",
            )
        with r2c2:
            epoch_sec = st.number_input(
                "Epoch length (s)",
                min_value=0.1,
                max_value=30.0,
                value=float(cfg.epoch_sec),
                step=0.1,
                key="epoch",
            )

        normalize = st.selectbox(
            "Normalization",
            options=["none", "z-score", "min-max"],
            index=["none", "z-score", "min-max"].index(
                cfg.normalize if cfg.normalize in ["none", "z-score", "min-max"] else "z-score"
            ),
            key="norm",
        )

        cfg.bandpass_low = _safe_float(bp_low, 0.5)
        cfg.bandpass_high = _safe_float(bp_high, 40.0)
        cfg.notch = _safe_float(notch, 50.0)
        cfg.epoch_sec = _safe_float(epoch_sec, 2.0)
        cfg.normalize = str(normalize)
        st.session_state.preproc = cfg

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Run preprocessing", use_container_width=True, disabled=(st.session_state.dataset_df is None)):
                run_pipeline()
        with b2:
            if st.button("Save config", use_container_width=True):
                save_run(action="save_config", status="Ready", metrics={"preproc": asdict(cfg)})

    with col2:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Current configuration</div>
                <div class="subtle">Stored in session + exported in runs/</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        st.json(asdict(cfg))

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="card">
              <div class="card-title">
                <div style="font-weight:800; font-size:1.05rem;">Notes</div>
                <div class="subtle">What this page controls</div>
              </div>
              <div class="small">
                - Filtering parameters (band-pass and notch)<br/>
                - Epoching window length<br/>
                - Data normalization strategy<br/>
                - Traceability: each run saves parameters + results
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def plot_cm(cm, title="Confusion Matrix"):
    if not MPL_OK:
        st.write(cm)
        return
    arr = np.array(cm, dtype=float)
    fig = plt.figure()
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
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    st.pyplot(fig, clear_figure=True)


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


ss_init()
sidebar_nav()
render_topbar()

page = st.session_state.page
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

if page == "Dashboard":
    render_dashboard()
elif page == "Import":
    render_import()
elif page == "Preprocess":
    render_preprocess()
elif page == "Results":
    render_results()
else:
    st.session_state.page = "Dashboard"
    render_dashboard()
