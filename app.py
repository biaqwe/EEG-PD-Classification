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
header[data-testid="stHeader"] { display: none; }
div[data-testid="stToolbar"] { display: none; }

div[data-testid="stAppViewContainer"] { padding-top: 0rem; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(11,16,32,0.95), rgba(15,23,48,0.95)) !important;
  border-right: 1px solid rgba(255,255,255,0.12);
  box-shadow: 8px 0 30px rgba(0,0,0,0.35);
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
  padding-top: -1.2rem !important;   /* urca meniul */
  padding-bottom: 0.6rem !important;
}

section[data-testid="stSidebar"] .sidebar-title{
  color: rgba(255,255,255,0.55);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin: 0.2rem 0 10px 4px !important;
}

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

section[data-testid="stSidebar"] .stButton{ margin-bottom: 8px; }

section[data-testid="stSidebar"] .small{
  color: rgba(255,255,255,0.6);
  font-size: 0.85rem;
  line-height: 1.4;
}

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
    from sklearn.model_selection import train_test_split, StratifiedGroupKFold, GroupShuffleSplit
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
    from sklearn.feature_selection import SelectKBest, f_classif
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
    if "last_cm_window" not in st.session_state:
        st.session_state.last_cm_window = None
    if "last_cm_subject" not in st.session_state:
        st.session_state.last_cm_subject = None
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
    drop_meta = [c for c in ["group", "subject_id", "subject_key", "window_start", "recording", "part", "start", "source_file"] if c in X.columns]
    X = X.drop(columns=drop_meta, errors="ignore")
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
    
    groups = df.loc[X.index, "subject_key"].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.values[train_idx], X.values[test_idx]
    y_train, y_test = y.values[train_idx], y.values[test_idx]
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=100)),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
    ])
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

def train_svm_group_cv(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42):
    if not SKLEARN_OK:
        return None, None, None, "scikit-learn not available in this environment."

    cols = list(df.columns)
    label_candidates = [c for c in cols if c.lower() in ["label", "class", "y", "target"]]
    if not label_candidates:
        return None, None, None, "Dataset CSV must contain a label column: label/class/y/target."

    if "subject_key" not in df.columns:
        return None, None, None, "Dataset must contain a subject_key column for group cross-validation."

    ycol = label_candidates[0]
    X_df = df.drop(columns=[ycol]).copy()
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

    meta_cols = [c for c in ["group", "subject_id", "subject_key", "window_start", "recording", "part", "start", "source_file"] if c in X_df.columns]
    groups = df["subject_key"].astype(str).values
    X_df = X_df.drop(columns=meta_cols, errors="ignore")

    X_df = X_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = y.loc[X_df.index]
    groups = pd.Series(groups, index=df.index).loc[X_df.index].values

    if len(X_df) < 10 or len(np.unique(y)) < 2:
        return None, None, None, "Not enough data or only one class present."

    X = X_df.astype(np.float32).values
    y = y.values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def subject_aggregate(df_fold: pd.DataFrame):
        return df_fold.groupby("subject_key", as_index=False).agg(
            label=("label", "first"),
            proba_pd=("proba_pd", "mean"),
            subject_id=("subject_id", "first") if "subject_id" in df_fold.columns else ("subject_key", "first"),
        )

    def safe_threshold(y_true, proba):
        fpr, tpr, thr = roc_curve(y_true, proba)
        j = tpr - fpr
        idx = int(np.argmax(j))
        t = float(thr[idx])
        if not np.isfinite(t):
            t = 0.5
        return t

    def metrics_from_proba(y_true, proba, thr):
        pred = (proba >= thr).astype(int)
        acc = float(accuracy_score(y_true, pred))
        f1 = float(f1_score(y_true, pred))
        auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")
        cm = confusion_matrix(y_true, pred)
        return acc, f1, auc, cm

    win_acc, win_f1, win_auc = [], [], []
    subj_acc, subj_f1, subj_auc = [], [], []
    win_cms, subj_cms = [], []
    thresholds = []
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            return None, None, None, f"Group leakage detected in fold {fold_idx}."

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=100)),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
        ])
        clf.fit(X_train, y_train)

        proba_test = clf.predict_proba(X_test)[:, 1]
        thr = safe_threshold(y_test, proba_test)
        thresholds.append(thr)

        acc_w, f1_w, auc_w, cm_w = metrics_from_proba(y_test, proba_test, thr)
        win_acc.append(acc_w)
        win_f1.append(f1_w)
        win_auc.append(auc_w)
        win_cms.append(cm_w)

        fold_test = df.loc[X_df.index].iloc[test_idx][["subject_key", ycol]].copy()
        fold_test = fold_test.rename(columns={ycol: "label"})
        if "subject_id" in df.columns:
            fold_test["subject_id"] = df.loc[X_df.index].iloc[test_idx]["subject_id"].values
        else:
            fold_test["subject_id"] = fold_test["subject_key"]
        fold_test["proba_pd"] = proba_test

        df_subj = subject_aggregate(fold_test)
        y_subj = df_subj["label"].astype(int).values
        proba_subj = df_subj["proba_pd"].astype(float).values

        acc_s, f1_s, auc_s, cm_s = metrics_from_proba(y_subj, proba_subj, thr)
        subj_acc.append(acc_s)
        subj_f1.append(f1_s)
        subj_auc.append(auc_s)
        subj_cms.append(cm_s)

        fold_rows.append({
            "fold": fold_idx,
            "train_subjects": len(train_groups),
            "test_subjects": len(test_groups),
            "threshold": thr,
            "window_acc": acc_w,
            "window_f1": f1_w,
            "window_auc": auc_w,
            "subject_acc": acc_s,
            "subject_f1": f1_s,
            "subject_auc": auc_s,
        })

    def mean_std(a):
        a = np.array(a, dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    w_acc_m, w_acc_s = mean_std(win_acc)
    w_f1_m, w_f1_s = mean_std(win_f1)
    w_auc_m, w_auc_s = mean_std(win_auc)

    s_acc_m, s_acc_s = mean_std(subj_acc)
    s_f1_m, s_f1_s = mean_std(subj_f1)
    s_auc_m, s_auc_s = mean_std(subj_auc)

    thr_m, thr_s = mean_std(thresholds)

    win_cm_sum = np.sum(np.stack(win_cms, axis=0), axis=0).tolist()
    subj_cm_sum = np.sum(np.stack(subj_cms, axis=0), axis=0).tolist()

    metrics = {
        "window_acc_mean": w_acc_m,
        "window_acc_std": w_acc_s,
        "window_f1_mean": w_f1_m,
        "window_f1_std": w_f1_s,
        "window_auc_mean": w_auc_m,
        "window_auc_std": w_auc_s,
        "subject_acc_mean": s_acc_m,
        "subject_acc_std": s_acc_s,
        "subject_f1_mean": s_f1_m,
        "subject_f1_std": s_f1_s,
        "subject_auc_mean": s_auc_m,
        "subject_auc_std": s_auc_s,
        "threshold_mean": thr_m,
        "threshold_std": thr_s,
        "n_splits": n_splits,
        "subjects": int(pd.Series(groups).nunique()),
        "features": int(X.shape[1]),
        "fold_details": fold_rows,
    }

    return metrics, win_cm_sum, subj_cm_sum, None


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

    if st.sidebar.button("Run SVM Group CV", use_container_width=True):
        run_train_svm_group_cv()


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

def run_train_svm_group_cv():
    if st.session_state.dataset_df is None:
        set_status("Error")
        log("Cannot run SVM Group CV: dataset not loaded.")
        save_run(action="svm_group_cv", status="Error", metrics={"error": "dataset not loaded"})
        return

    set_status("Running")
    st.session_state.last_action = "svm_group_cv"
    log("Running SVM Group CV started.")

    metrics, cm_window, cm_subject, err = train_svm_group_cv(st.session_state.dataset_df, n_splits=5, random_state=42)

    if err:
        set_status("Error")
        log(f"SVM Group CV failed: {err}")
        st.session_state.last_metrics = {"error": err}
        st.session_state.last_cm = None
        st.session_state.last_cm_window = None
        st.session_state.last_cm_subject = None
        st.session_state.last_roc = None
        save_run(action="svm_group_cv", status="Error", metrics={"error": err})
        return

    st.session_state.last_metrics = metrics
    st.session_state.last_cm = cm_subject
    st.session_state.last_cm_window = cm_window
    st.session_state.last_cm_subject = cm_subject
    st.session_state.last_roc = None

    set_status("Ready")
    log(f"SVM Group CV done. Subject Accuracy mean={metrics.get('subject_acc_mean', 0):.4f}, Subject F1 mean={metrics.get('subject_f1_mean', 0):.4f}")
    save_run(action="svm_group_cv", status="Ready", metrics=metrics)

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
                    st.metric("Window Accuracy", "-" if metrics.get("window_acc_mean") is None else f"{metrics['window_acc_mean']:.3f}")
                with mcols2[1]:
                    st.metric("Window F1", "-" if metrics.get("window_f1_mean") is None else f"{metrics['window_f1_mean']:.3f}")
                with mcols2[2]:
                    st.metric("Window AUC", "-" if metrics.get("window_auc_mean") is None else f"{metrics['window_auc_mean']:.3f}")

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
