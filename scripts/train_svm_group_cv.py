import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

CSV_PATH = "dataset_iowa_pd_hc.csv"
N_SPLITS = 5
RANDOM_STATE = 42

def subject_aggregate(df_fold: pd.DataFrame):
    g = df_fold.groupby("subject_key", as_index=False).agg(
        label=("label", "first"),
        proba_pd=("proba_pd", "mean"),
        subject_id=("subject_id", "first"),
    )
    return g

def find_best_threshold(y_true: np.ndarray, proba: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx])

def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, thr: float):
    pred = (proba >= thr).astype(int)
    acc = float(accuracy_score(y_true, pred))
    f1 = float(f1_score(y_true, pred))
    auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else np.nan
    cm = confusion_matrix(y_true, pred)
    return acc, f1, auc, cm

def mean_std(x):
    x = np.array(x, dtype=float)
    return float(np.nanmean(x)), float(np.nanstd(x))

def main():
    df = pd.read_csv(CSV_PATH)

    required = {"label", "subject_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    meta_cols = {"label", "group", "subject_id", "subject_key", "window_start"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].astype(np.float32).values
    y = df["label"].astype(int).values
    groups = df["subject_key"].astype(str).values

    print("Rows:", df.shape[0], "Features:", len(feature_cols))
    print("Subjects:", df["subject_key"].nunique())
    print("Label counts:\n", df["label"].value_counts())

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    win_acc, win_f1, win_auc = [], [], []
    subj_acc, subj_f1, subj_auc = [], [], []
    win_cms = []
    subj_cms = []

    thresholds = []

    fold = 0
    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        fold += 1

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(f"Group leakage detected in fold {fold}: {len(overlap)} overlapping subjects")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
        clf.fit(X_train, y_train)

        proba_test = clf.predict_proba(X_test)[:, 1]

        thr = find_best_threshold(y_test, proba_test)
        thresholds.append(thr)

        acc_w, f1_w, auc_w, cm_w = metrics_from_proba(y_test, proba_test, thr)
        win_acc.append(acc_w); win_f1.append(f1_w); win_auc.append(auc_w)
        win_cms.append(cm_w)

        df_test = df.iloc[test_idx][["subject_key", "subject_id", "label"]].copy()
        df_test["proba_pd"] = proba_test

        df_subj = subject_aggregate(df_test)
        y_subj = df_subj["label"].astype(int).values
        proba_subj = df_subj["proba_pd"].astype(float).values

        acc_s, f1_s, auc_s, cm_s = metrics_from_proba(y_subj, proba_subj, thr)
        subj_acc.append(acc_s); subj_f1.append(f1_s); subj_auc.append(auc_s)
        subj_cms.append(cm_s)

        print(f"\n--- Fold {fold}/{N_SPLITS} ---")
        print("Train subjects:", len(train_groups), "Test subjects:", len(test_groups), "Overlap:", len(overlap))
        print("Threshold (Youden J):", round(thr, 4))
        print("Window-level  | Acc:", round(acc_w, 4), "F1:", round(f1_w, 4), "AUC:", round(auc_w, 4))
        print("Subject-level | Acc:", round(acc_s, 4), "F1:", round(f1_s, 4), "AUC:", round(auc_s, 4))

    w_acc_m, w_acc_s = mean_std(win_acc)
    w_f1_m, w_f1_s = mean_std(win_f1)
    w_auc_m, w_auc_s = mean_std(win_auc)

    s_acc_m, s_acc_s = mean_std(subj_acc)
    s_f1_m, s_f1_s = mean_std(subj_f1)
    s_auc_m, s_auc_s = mean_std(subj_auc)

    thr_m, thr_s = mean_std(thresholds)

    win_cm_sum = np.sum(np.stack(win_cms, axis=0), axis=0)
    subj_cm_sum = np.sum(np.stack(subj_cms, axis=0), axis=0)

    print("\n==============================")
    print("Subject-stratified Group CV results")
    print("==============================")
    print(f"Threshold (mean±std): {thr_m:.4f} ± {thr_s:.4f}")

    print("\nWindow-level (mean±std):")
    print(f"  Accuracy: {w_acc_m:.4f} ± {w_acc_s:.4f}")
    print(f"  F1:       {w_f1_m:.4f} ± {w_f1_s:.4f}")
    print(f"  AUC:      {w_auc_m:.4f} ± {w_auc_s:.4f}")
    print("  Confusion matrix (summed over folds):")
    print(win_cm_sum)

    print("\nSubject-level (mean±std):")
    print(f"  Accuracy: {s_acc_m:.4f} ± {s_acc_s:.4f}")
    print(f"  F1:       {s_f1_m:.4f} ± {s_f1_s:.4f}")
    print(f"  AUC:      {s_auc_m:.4f} ± {s_auc_s:.4f}")
    print("  Confusion matrix (summed over folds):")
    print(subj_cm_sum)

    results = {
        "n_splits": N_SPLITS,
        "threshold_mean": thr_m,
        "threshold_std": thr_s,
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
        "window_cm_sum": win_cm_sum.tolist(),
        "subject_cm_sum": subj_cm_sum.tolist(),
    }
    pd.Series(results).to_json("svm_group_cv_results.json", indent=2)
    print("\nSaved: svm_group_cv_results.json")

if __name__ == "__main__":
    main()
