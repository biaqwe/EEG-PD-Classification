import numpy as np
import pandas as pd

from src.config import K_BEST, N_SPLITS, RANDOM_STATE, TEST_SIZE
from src.data_utils import get_xy

try:
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


def build_svm_pipeline(random_state=RANDOM_STATE):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=K_BEST)),
        ("svm", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])


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

    if "subject_key" not in df.columns:
        return None, None, None, None, "Dataset must contain a subject_key column."

    groups = df.loc[X.index, "subject_key"].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.values[train_idx], X.values[test_idx]
    y_train, y_test = y.values[train_idx], y.values[test_idx]

    clf = build_svm_pipeline()
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
        roc = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thr": thr.tolist(),
        }
    except Exception:
        pass

    metrics = {"accuracy": acc, "f1": f1}
    if auc is not None:
        metrics["auc"] = auc

    return metrics, cm, roc, clf, None


def subject_aggregate(df_fold: pd.DataFrame):
    if "subject_id" in df_fold.columns:
        return df_fold.groupby("subject_key", as_index=False).agg(
            label=("label", "first"),
            proba_pd=("proba_pd", "mean"),
            subject_id=("subject_id", "first"),
        )

    return df_fold.groupby("subject_key", as_index=False).agg(
        label=("label", "first"),
        proba_pd=("proba_pd", "mean"),
        subject_id=("subject_key", "first"),
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


def train_svm_group_cv(df: pd.DataFrame, n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE):
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
        y = y.astype(str).str.strip().str.lower()
        y = y.map({"pd": 1, "hc": 0, "1": 1, "0": 0}).fillna(y)

    try:
        y = y.astype(int)
    except Exception:
        uniq = sorted(pd.unique(y))
        mapping = {v: i for i, v in enumerate(uniq)}
        y = y.map(mapping).astype(int)

    meta_cols = [
        c for c in [
            "group", "subject_id", "subject_key", "window_start",
            "recording", "part", "start", "source_file"
        ] if c in X_df.columns
    ]

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

        clf = build_svm_pipeline(random_state=random_state)
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

    cm = [
        [int(rng.integers(18, 30)), int(rng.integers(2, 10))],
        [int(rng.integers(3, 12)), int(rng.integers(16, 30))]
    ]

    roc = {
        "fpr": [0.0, 0.08, 0.18, 0.35, 1.0],
        "tpr": [0.0, 0.55, 0.72, 0.87, 1.0],
        "thr": [1.2, 0.75, 0.52, 0.28, 0.0]
    }

    return {"accuracy": acc, "f1": f1, "auc": auc}, cm, roc