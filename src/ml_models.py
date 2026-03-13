import copy

import numpy as np
import pandas as pd

from src.config import (
    K_BEST,
    N_SPLITS,
    RANDOM_STATE,
    TEST_SIZE,
    RAW_WINDOW_SEC,
    RAW_STEP_SEC,
    RAW_MAX_WINDOWS_PER_RECORDING,
    RAW_L_FREQ,
    RAW_H_FREQ,
    RAW_NOTCH_FREQ,
    RAW_USE_BANDPASS,
    RAW_USE_NOTCH,
    RAW_VAL_SIZE,
    CNN_EPOCHS,
    CNN_BATCH_SIZE,
    CNN_LR,
    CNN_WEIGHT_DECAY,
    CNN_DROPOUT,
    CNN_PATIENCE,
)
from src.data_utils import get_xy
from src.raw_eeg import load_brainvision_windows

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

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_OK = True
except Exception:
    TORCH_OK = False


def build_svm_pipeline(k_best=100, random_state=RANDOM_STATE):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k_best)),
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
        roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": thr.tolist()}
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
        return None, None, None, None, "scikit-learn not available in this environment."

    cols = list(df.columns)
    label_candidates = [c for c in cols if c.lower() in ["label", "class", "y", "target"]]
    if not label_candidates:
        return None, None, None, None, "Dataset CSV must contain a label column: label/class/y/target."

    if "subject_key" not in df.columns:
        return None, None, None, None, "Dataset must contain a subject_key column for group cross-validation."

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
        return None, None, None, None, "Not enough data or only one class present."

    X = X_df.astype(np.float32).values
    y = y.values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    win_acc, win_f1, win_auc = [], [], []
    subj_acc, subj_f1, subj_auc = [], [], []
    win_cms, subj_cms = [], []
    thresholds = []
    fold_rows = []
    sample_prediction_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            return None, None, None, None, f"Group leakage detected in fold {fold_idx}."

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        k_best = min(K_BEST, X_train.shape[1])
        clf = build_svm_pipeline(k_best=k_best, random_state=random_state)
        clf.fit(X_train, y_train)

        proba_test = clf.predict_proba(X_test)[:, 1]
        thr = safe_threshold(y_test, proba_test)
        thresholds.append(thr)

        pred_test = (proba_test >= thr).astype(int)

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

        test_original_rows = df.loc[X_df.index].iloc[test_idx].copy()

        for local_i in range(len(test_idx)):
            row_info = {
                "row_index": int(test_original_rows.index[local_i]),
                "fold": int(fold_idx),
                "subject_key": str(test_original_rows.iloc[local_i]["subject_key"]) if "subject_key" in test_original_rows.columns else "",
                "subject_id": str(test_original_rows.iloc[local_i]["subject_id"]) if "subject_id" in test_original_rows.columns else "",
                "window_start": int(test_original_rows.iloc[local_i]["window_start"]) if "window_start" in test_original_rows.columns else -1,
                "true_label": int(y_test[local_i]),
                "pred_label": int(pred_test[local_i]),
                "proba_pd": float(proba_test[local_i]),
                "proba_hc": float(1.0 - proba_test[local_i]),
                "threshold": float(thr),
            }
            sample_prediction_rows.append(row_info)

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

    sample_predictions_df = (
        pd.DataFrame(sample_prediction_rows)
        .sort_values("row_index")
        .reset_index(drop=True)
    )

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

    return metrics, win_cm_sum, subj_cm_sum, sample_predictions_df, None


class EEGNetLite(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, dropout: float = 0.25):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 8)),
        )

        self.classifier = nn.Linear(16 * 1 * 8, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze(1)


def _normalize_by_train(X_train, X_val, X_test):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_val, X_test


def train_raw_eeg_cnn(payloads: dict, config: dict | None = None):
    if not SKLEARN_OK:
        return None, None, None, None, None, "scikit-learn not available in this environment."

    if not TORCH_OK:
        return None, None, None, None, None, "PyTorch is not available in this environment."

    cfg = config or {}
    window_sec = float(cfg.get("window_sec", RAW_WINDOW_SEC))
    step_sec = float(cfg.get("step_sec", RAW_STEP_SEC))
    max_windows = int(cfg.get("max_windows_per_recording", RAW_MAX_WINDOWS_PER_RECORDING))
    use_bandpass = bool(cfg.get("use_bandpass", RAW_USE_BANDPASS))
    l_freq = float(cfg.get("bandpass_low", RAW_L_FREQ))
    h_freq = float(cfg.get("bandpass_high", RAW_H_FREQ))
    use_notch = bool(cfg.get("use_notch", RAW_USE_NOTCH))
    notch_freq = float(cfg.get("notch_freq", RAW_NOTCH_FREQ))

    X, y, groups, meta_df, summary, err = load_brainvision_windows(
        payloads=payloads,
        window_sec=window_sec,
        step_sec=step_sec,
        max_windows_per_recording=max_windows,
        use_bandpass=use_bandpass,
        l_freq=l_freq,
        h_freq=h_freq,
        use_notch=use_notch,
        notch_freq=notch_freq,
    )
    if err:
        return None, None, None, None, None, err

    if len(np.unique(y)) < 2:
        return None, None, None, None, None, "Need both classes (HC and PD) in the uploaded recordings."

    gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss_test.split(X, y, groups))

    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    groups_trainval = groups[trainval_idx]

    val_fraction = RAW_VAL_SIZE
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=RANDOM_STATE)
    train_idx_local, val_idx_local = next(gss_val.split(X_trainval, y_trainval, groups_trainval))

    X_train, X_val = X_trainval[train_idx_local], X_trainval[val_idx_local]
    y_train, y_val = y_trainval[train_idx_local], y_trainval[val_idx_local]

    X_train, X_val, X_test = _normalize_by_train(X_train, X_val, X_test)

    X_train = X_train[:, None, :, :]
    X_val = X_val[:, None, :, :]
    X_test = X_test[:, None, :, :]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    model = EEGNetLite(
        n_channels=X_train.shape[2],
        n_samples=X_train.shape[3],
        dropout=CNN_DROPOUT,
    ).to(device)

    pos_count = max(1, int((y_train == 1).sum()))
    neg_count = max(1, int((y_train == 0).sum()))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CNN_LR,
        weight_decay=CNN_WEIGHT_DECAY,
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=CNN_BATCH_SIZE,
        shuffle=True,
    )

    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_state = None
    best_val_loss = float("inf")
    patience_left = CNN_PATIENCE

    for _ in range(CNN_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = CNN_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(test_x)
        proba = torch.sigmoid(logits).cpu().numpy()

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

    test_meta = meta_df.iloc[test_idx].copy().reset_index(drop=True)
    test_meta["true_label"] = y_test.astype(int)
    test_meta["pred_label"] = pred.astype(int)
    test_meta["proba_pd"] = proba.astype(float)
    test_meta["proba_hc"] = (1.0 - proba).astype(float)

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "n_windows_total": int(summary["n_windows"]),
        "n_windows_train": int(len(X_train)),
        "n_windows_val": int(len(X_val)),
        "n_windows_test": int(len(X_test)),
        "n_subjects": int(summary["n_subjects"]),
        "n_recordings": int(summary["n_recordings"]),
        "n_channels": int(summary["n_channels"]),
        "window_samples": int(summary["window_samples"]),
        "sampling_rate": float(summary["sampling_rate"]) if summary["sampling_rate"] is not None else None,
        "model": "EEGNetLite",
    }

    return metrics, cm, roc, model, test_meta, None