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
    SPECTROGRAM_HEIGHT,
    SPECTROGRAM_WIDTH,
)
from src.data_utils import get_xy
from src.raw_eeg import load_brainvision_spectrograms

try:
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        recall_score,
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
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    TORCH_OK = True
except Exception:
    TORCH_OK = False


def _torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _balanced_sampler(y):
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[y]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def build_svm_pipeline(k_best=100, random_state=RANDOM_STATE): # creates pipeline for training svm
    return Pipeline([
        ("scaler", StandardScaler()), # transforms each feat to have mean=0 and sd=1 bcs svm works better when feats are on similar scales
        ("select", SelectKBest(score_func=f_classif, k=k_best)), # selects most important feats to reduce noise and improve training
        ("svm", SVC(
            kernel="rbf", # uses radial basis func kernel for svm to separate complex data patterns
            probability=True, # allows the model to compute class probabilities
            class_weight="balanced", # if a class has fewer samples svm aadjusts the weights
            random_state=random_state, # sets fixed random seed for reproducible results
        )),
    ])


def train_svm(df: pd.DataFrame): # trains and evaluates svm model
    if not SKLEARN_OK:
        return None, None, None, None, "scikit-learn not available in this environment."

    # extracts feats and labels
    X, y, err = get_xy(df)
    if err:
        return None, None, None, None, err

    # cleans invalid numeric values
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = y.loc[X.index]

    # checks if enough usable data
    if len(X) < 10 or len(np.unique(y)) < 2:
        return None, None, None, None, "Not enough data or only one class present."

    # checks for subject_key col
    if "subject_key" not in df.columns:
        return None, None, None, None, "Dataset must contain a subject_key column."

    groups = df.loc[X.index, "subject_key"].astype(str).values # gets the group label for each sample

    # creates a group aware train/test split, samples from the same subject are kept together
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = sorted(train_groups.intersection(test_groups))

    if overlap:
        return None, None, None, None, (
            "Subject leakage detected in SVM train/test split: " + ", ".join(overlap)
        )

    # builds train and test sets
    X_train, X_test = X.values[train_idx], X.values[test_idx]
    y_train, y_test = y.values[train_idx], y.values[test_idx]

    clf = build_svm_pipeline() # creates the ml pipeline
    clf.fit(X_train, y_train) # trains the model using the training data

    proba = clf.predict_proba(X_test)[:, 1] # predicts the probability of pd for each test sample
    pred = (proba >= 0.5).astype(int) # turns probabilities into class lables

    # basic evaluation metrics
    acc = float(accuracy_score(y_test, pred)) # how many predictions were correct
    f1 = float(f1_score(y_test, pred)) # balance between precision and recall
    cm = confusion_matrix(y_test, pred).tolist() # confusion matrix

    # computes auc and roc
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


def subject_aggregate(df_fold: pd.DataFrame): # combines multiple window predictions from the same subject into a single subject level prediction
    if "subject_id" in df_fold.columns:
        # group rows by subject
        return df_fold.groupby("subject_key", as_index=False).agg(
            label=("label", "first"), # true label
            proba_pd=("proba_pd", "mean"), # avg probability across windows
            subject_id=("subject_id", "first"), # subject id
        )

    return df_fold.groupby("subject_key", as_index=False).agg(
        label=("label", "first"),
        proba_pd=("proba_pd", "mean"),
        subject_id=("subject_key", "first"),
    )


def safe_threshold(y_true, proba):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5

    candidates = np.unique(
        np.concatenate([
            np.array([0.5]),
            np.linspace(0.35, 0.80, 91),
            proba,
        ])
    )

    best_threshold = 0.5
    best_score = -1e9

    for threshold in candidates:
        threshold = float(np.clip(threshold, 0.35, 0.85))
        pred = (proba >= threshold).astype(int)

        if len(np.unique(pred)) < 2:
            continue

        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bacc = balanced_accuracy_score(y_true, pred)
        acc = accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, zero_division=0)

        score = (0.65 * bacc) + (0.25 * acc) + (0.10 * f1) - (0.10 * abs(sensitivity - specificity))

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold)


def metrics_from_proba(y_true, proba, thr): # calculates metrics
    pred = (proba >= thr).astype(int) # converts probabilities intro predictied classes
    acc = float(accuracy_score(y_true, pred)) # accuracy
    f1 = float(f1_score(y_true, pred, zero_division=0)) # f1 score
    auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan") # area under the roc curve
    cm = confusion_matrix(y_true, pred, labels=[0, 1]) # confusion matrix
    return acc, f1, auc, cm

def train_svm_group_cv(df: pd.DataFrame, n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE): # trains and evaluates svm model using gorup cross validation
    if not SKLEARN_OK:
        return None, None, None, None, "scikit-learn not available in this environment."

    # checks for class label col
    cols = list(df.columns)
    label_candidates = [c for c in cols if c.lower() in ["label", "class", "y", "target"]]
    if not label_candidates:
        return None, None, None, None, "Dataset CSV must contain a label column: label/class/y/target."

    # checks for subject_key col
    if "subject_key" not in df.columns:
        return None, None, None, None, "Dataset must contain a subject_key column for group cross-validation."

    # chooses the first detected col label and separates the data into feats and metadata and target labels
    ycol = label_candidates[0]
    X_df = df.drop(columns=[ycol]).copy()
    y = df[ycol].copy()

    # converst text lables to nrs
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower()
        y = y.map({"pd": 1, "hc": 0, "1": 1, "0": 0}).fillna(y)

    # makes sure all labels are ints
    try:
        y = y.astype(int)
    except Exception:
        uniq = sorted(pd.unique(y))
        mapping = {v: i for i, v in enumerate(uniq)}
        y = y.map(mapping).astype(int)

    # identifies metadata cols that shouldnt be used for training
    meta_cols = [
        c for c in [
            "group", "subject_id", "subject_key", "window_start",
            "recording", "part", "start", "source_file"
        ] if c in X_df.columns
    ]

    # creates the group labels for cv
    groups = df["subject_key"].astype(str).values
    # removes metadata cols from feats
    X_df = X_df.drop(columns=meta_cols, errors="ignore")

    # cleans invalid numeric values
    X_df = X_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = y.loc[X_df.index]
    groups = pd.Series(groups, index=df.index).loc[X_df.index].values

    # checks if enough usable data
    if len(X_df) < 10 or len(np.unique(y)) < 2:
        return None, None, None, None, "Not enough data or only one class present."

    # converts feats and lables to numpy arrays
    X = X_df.astype(np.float32).values
    y = y.values

    # creates the cv splitter
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # lists for fold results
    win_acc, win_f1, win_auc = [], [], [] # window level metrics
    subj_acc, subj_f1, subj_auc = [], [], [] # subject level metrics
    win_cms, subj_cms = [], [] # confusion matrices
    thresholds = [] # thresholds
    fold_rows = [] # fold summary info
    sample_prediction_rows = [] # individual sample predictions

    # runs cv fold by fold
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        # checks that the same subject doesnt appear in both train and test
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            return None, None, None, None, f"Group leakage detected in fold {fold_idx}."

        # builds train and test data for the current fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # makes sure that nr of selected feats <= nr of avail feats
        k_best = min(K_BEST, X_train.shape[1])
        clf = build_svm_pipeline(k_best=k_best, random_state=random_state) # creates the ml pipeline
        clf.fit(X_train, y_train) # trains the model using the training data of current fold

        proba_test = clf.predict_proba(X_test)[:, 1] # predicts the probability of pd for each test sample
        # fixed threshold avoids using the test fold to choose the threshold
        thr = 0.5
        thresholds.append(thr)

        pred_test = (proba_test >= thr).astype(int) # turns probabilities into class lables

        # computes window level metrics
        acc_w, f1_w, auc_w, cm_w = metrics_from_proba(y_test, proba_test, thr)
        win_acc.append(acc_w) # window accuracy
        win_f1.append(f1_w) # window f1 score
        win_auc.append(auc_w) # window auc
        win_cms.append(cm_w) # window confusion matrix

        # creates a dataframe for the current test fold
        fold_test = df.loc[X_df.index].iloc[test_idx][["subject_key", ycol]].copy()
        fold_test = fold_test.rename(columns={ycol: "label"})

        # if dataset contains subject_id its used
        if "subject_id" in df.columns:
            fold_test["subject_id"] = df.loc[X_df.index].iloc[test_idx]["subject_id"].values
        else:
            fold_test["subject_id"] = fold_test["subject_key"]

        # adds the predicted probability for each window in fold
        fold_test["proba_pd"] = proba_test

        # all window level predictions from the same subject are averaged into one subject level prediction
        df_subj = subject_aggregate(fold_test)
        y_subj = df_subj["label"].astype(int).values
        proba_subj = df_subj["proba_pd"].astype(float).values

        # computes subject level metrics
        acc_s, f1_s, auc_s, cm_s = metrics_from_proba(y_subj, proba_subj, thr)
        subj_acc.append(acc_s) # subject accuracy
        subj_f1.append(f1_s) # subject f1 score
        subj_auc.append(auc_s) # subject auc
        subj_cms.append(cm_s) # subject confusion matrix

        # gets original rows of test fold
        test_original_rows = df.loc[X_df.index].iloc[test_idx].copy()

        # stores detailed prediction record for each test row
        for local_i in range(len(test_idx)):
            row_info = {
                "row_index": int(test_original_rows.index[local_i]), # original row index
                "fold": int(fold_idx), # fold nr
                "subject_key": str(test_original_rows.iloc[local_i]["subject_key"]) if "subject_key" in test_original_rows.columns else "", # subject info
                "subject_id": str(test_original_rows.iloc[local_i]["subject_id"]) if "subject_id" in test_original_rows.columns else "", # subject info
                "window_start": int(test_original_rows.iloc[local_i]["window_start"]) if "window_start" in test_original_rows.columns else -1, # window start
                "true_label": int(y_test[local_i]), # true label
                "pred_label": int(pred_test[local_i]), # predicted label
                "proba_pd": float(proba_test[local_i]), # probabilities
                "proba_hc": float(1.0 - proba_test[local_i]), # probabilities
                "threshold": float(thr), # threshold used
            }
            sample_prediction_rows.append(row_info)

        # stores summary for current fold
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

    # helper for mean and std
    def mean_std(a):
        a = np.array(a, dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    # mean and std for window level metrics
    w_acc_m, w_acc_s = mean_std(win_acc)
    w_f1_m, w_f1_s = mean_std(win_f1)
    w_auc_m, w_auc_s = mean_std(win_auc)

    # mean and std for subject level metrics
    s_acc_m, s_acc_s = mean_std(subj_acc)
    s_f1_m, s_f1_s = mean_std(subj_f1)
    s_auc_m, s_auc_s = mean_std(subj_auc)

    # mean and std for thresholds
    thr_m, thr_s = mean_std(thresholds)

    # combines all fold cms into one total cm, one window level one subject level
    win_cm_sum = np.sum(np.stack(win_cms, axis=0), axis=0).tolist()
    subj_cm_sum = np.sum(np.stack(subj_cms, axis=0), axis=0).tolist()

    # builds sample predictions
    sample_predictions_df = (
        pd.DataFrame(sample_prediction_rows)
        .sort_values("row_index")
        .reset_index(drop=True)
    )

    # final metrics
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


def _stratified_group_split(X, y, groups, n_splits=4, random_state=RANDOM_STATE, fold_index=0): # creates a stratified subject-level split
    subject_df = pd.DataFrame({
        "subject_key": groups,
        "label": y,
    }).groupby("subject_key", as_index=False).agg(
        label=("label", "first")
    )

    min_groups_per_class = int(subject_df["label"].value_counts().min())
    effective_splits = min(int(n_splits), min_groups_per_class)

    if effective_splits >= 2:
        splitter = StratifiedGroupKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(X, y, groups=groups))
        return splits[int(fold_index) % len(splits)]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=random_state,
    )
    return next(splitter.split(X, y, groups))


def _split_info(y, groups, idx, prefix): # returns class distribution for one split
    y_part = np.asarray(y)[idx]
    groups_part = np.asarray(groups)[idx]

    subject_df = pd.DataFrame({
        "subject_key": groups_part,
        "label": y_part,
    }).groupby("subject_key", as_index=False).agg(
        label=("label", "first")
    )

    return {
        f"{prefix}_windows": int(len(y_part)),
        f"{prefix}_subjects": int(subject_df["subject_key"].nunique()),
        f"{prefix}_hc_windows": int((y_part == 0).sum()),
        f"{prefix}_pd_windows": int((y_part == 1).sum()),
        f"{prefix}_hc_subjects": int((subject_df["label"] == 0).sum()),
        f"{prefix}_pd_subjects": int((subject_df["label"] == 1).sum()),
    }


def _subjects_with_multiple_labels(y, groups): # checks if one subject has both labels
    df = pd.DataFrame({
        "subject_key": np.asarray(groups).astype(str),
        "label": np.asarray(y).astype(int),
    })

    label_counts = df.groupby("subject_key")["label"].nunique()
    bad_subjects = label_counts[label_counts > 1].index.tolist()
    return bad_subjects


def _check_subject_leakage(groups, split_indices): # checks if one subject appears in more than one split
    groups = np.asarray(groups).astype(str)

    split_subjects = {}
    for split_name, indices in split_indices.items():
        split_subjects[split_name] = set(groups[indices])

    overlaps = []

    split_names = list(split_subjects.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a = split_names[i]
            b = split_names[j]
            common = sorted(split_subjects[a].intersection(split_subjects[b]))

            if common:
                overlaps.append({
                    "split_a": a,
                    "split_b": b,
                    "subjects": common,
                })

    return overlaps, split_subjects


def _subject_list_metrics(split_subjects): # stores subject lists in metrics
    out = {}

    for split_name, subjects in split_subjects.items():
        out[f"{split_name}_subject_keys"] = ", ".join(sorted(subjects))

    return out


def _binary_extra_metrics(y_true, pred): # computes extra binary classification metrics
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def _probability_diagnostics(y_true, proba, prefix): # checks if pd probabilities are higher for real pd samples
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    hc_mask = y_true == 0
    pd_mask = y_true == 1

    mean_hc = float(np.mean(proba[hc_mask])) if hc_mask.any() else None
    mean_pd = float(np.mean(proba[pd_mask])) if pd_mask.any() else None

    auc = None
    auc_if_flipped = None
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, proba))
        auc_if_flipped = float(roc_auc_score(y_true, 1.0 - proba))

    return {
        f"{prefix}_mean_proba_pd_for_true_hc": mean_hc,
        f"{prefix}_mean_proba_pd_for_true_pd": mean_pd,
        f"{prefix}_auc": auc,
        f"{prefix}_auc_if_flipped": auc_if_flipped,
        f"{prefix}_looks_inverted": bool(mean_pd is not None and mean_hc is not None and mean_pd < mean_hc),
    }


class EEGNetLite(nn.Module): # defines a neural network mode for eeg classification
    def __init__(self, n_channels: int, n_samples: int, dropout: float = 0.25): # nr of eeg channels, nr of time samples, dropout probability
        super().__init__()

        self.block1 = nn.Sequential( # extracts spatial and temporal feats from eeg sig
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False), # detects paterns across time
            # 1: input channel, 8: nr of filters, (1,64): temporal kernel, padding: keeps sig lenght
            nn.BatchNorm2d(8), # normalizes feat maps to stabilize trainign

            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False), # detects spatial relationships
            # 8 groups so each filter processes channels separately
            nn.BatchNorm2d(16), # another normalization
            nn.ELU(), # exponential linear unit for complet patterns
            nn.AvgPool2d(kernel_size=(1, 4)), # reduces temporal dimension
            nn.Dropout(dropout), # disables neurons randomly during training to prevent overfitiing and improve generalization
        )

        self.block2 = nn.Sequential( # extracts higher level eeg feats
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False), # additional temporal filters
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False), # mixes feats across channels
            nn.BatchNorm2d(16), # stabilizes training
            nn.ELU(), # exponential linear unit for complet patterns
            nn.AvgPool2d(kernel_size=(1, 8)), # reduces temporal dimension further
            nn.Dropout(dropout), # disables neurons randomly during training to prevent overfitiing and improve generalization
            nn.AdaptiveAvgPool2d((1, 8)), # forces output size to alwas be 1 x 8
        )

        self.classifier = nn.Linear(16 * 1 * 8, 1) # final fully connected layer
        # input: 16 feat maps x 1 x 8
        # output: 1 neuron

    def forward(self, x): # defines how data flows through network
        x = self.block1(x) # extracts first level temporal and spatial feats
        x = self.block2(x) # extracts deeper eeg feats
        x = torch.flatten(x, 1) # tranforms feat maps into 1d vector
        x = self.classifier(x) # final prediction score
        return x.squeeze(1) # removes unnecessary dimesnions for output to be a simple vector


class SpectrogramCNN(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.35):
        super().__init__()

        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),

            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


def _normalize_by_train(X_train, X_val, X_test): # normalizes training, validation and test data
    if X_train.ndim == 3:
        mean = X_train.mean(axis=(0, 2), keepdims=True) # avg value of training data per eeg channel
        std = X_train.std(axis=(0, 2), keepdims=True) # std of training data per eeg channel
    else:
        mean = X_train.mean(axis=(0, 2, 3), keepdims=True) # avg value of training spectrograms per eeg channel
        std = X_train.std(axis=(0, 2, 3), keepdims=True) # std of training spectrograms per eeg channel

    std[std < 1e-8] = 1.0 # avoids /0

    X_train = (X_train - mean) / std # z score normalization
    X_val = (X_val - mean) / std # same training avg and std to not leak info
    X_test = (X_test - mean) / std # same training stats
    return X_train, X_val, X_test


def _predict_proba(model, X, device): # runs cnn prediction and returns pd probabilities
    model.eval() # prediction mode
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32).to(device) # converts data to tensor
        logits = model(x) # runs model
        proba = torch.sigmoid(logits).cpu().numpy() # PD probabilities
    return proba


def train_raw_eeg_cnn(payloads: dict, config: dict | None = None): # trains and evaluates cnn model
    # payloads: uploaded files, config: preprocessing settings
    if not SKLEARN_OK:
        return None, None, None, None, None, "scikit-learn not available in this environment."

    if not TORCH_OK:
        return None, None, None, None, None, "PyTorch is not available in this environment."

    # loads preprocessing settingss
    cfg = config or {}
    window_sec = float(cfg.get("window_sec", RAW_WINDOW_SEC)) # window length
    step_sec = float(cfg.get("step_sec", RAW_STEP_SEC)) # step size
    max_windows = int(cfg.get("max_windows_per_recording", RAW_MAX_WINDOWS_PER_RECORDING)) # max windows per recorfing
    # bandpass filter
    use_bandpass = bool(cfg.get("use_bandpass", RAW_USE_BANDPASS))
    l_freq = float(cfg.get("bandpass_low", RAW_L_FREQ))
    h_freq = float(cfg.get("bandpass_high", RAW_H_FREQ))
    # nothc filter
    use_notch = bool(cfg.get("use_notch", RAW_USE_NOTCH))
    notch_freq = float(cfg.get("notch_freq", RAW_NOTCH_FREQ))
    visual_height = int(cfg.get("visual_height", SPECTROGRAM_HEIGHT)) # spectrogram height
    visual_width = int(cfg.get("visual_width", SPECTROGRAM_WIDTH)) # spectrogram width

    spectrogram_config = {
        "window_sec": window_sec,
        "step_sec": step_sec,
        "max_windows_per_recording": max_windows,
        "use_bandpass": use_bandpass,
        "bandpass_low": l_freq,
        "bandpass_high": h_freq,
        "use_notch": use_notch,
        "notch_freq": notch_freq,
        "visual_height": visual_height,
        "visual_width": visual_width,
        "spectrogram_channel_mode": str(cfg.get("spectrogram_channel_mode", "mean")),
        "spectrogram_nperseg": int(cfg.get("spectrogram_nperseg", 128)),
        "spectrogram_overlap_ratio": float(cfg.get("spectrogram_overlap_ratio", 0.75)),
    }

    # turns raw eeg files into spectrogram images for model
    X, y, groups, meta_df, summary, err = load_brainvision_spectrograms(
    # X: spectrogram images, y: window labels, groups: subject or group ids, meta_df: window metadata, summary: info about ddataset, err: error msg
        payloads=payloads,
        config=spectrogram_config,
    )
    if err:
        return None, None, None, None, None, err

    # cnn can only learn if both PD and HC are present so checks if data includes them both
    if len(np.unique(y)) < 2:
        return None, None, None, None, None, "Need both classes (HC and PD) in the uploaded recordings."

    bad_subjects = _subjects_with_multiple_labels(y, groups)
    if bad_subjects:
        return None, None, None, None, None, (
            "Subject label inconsistency detected. These subjects have both HC and PD labels: "
            + ", ".join(bad_subjects)
        )

    # creates a stratified subject-level test split
    trainval_idx, test_idx = _stratified_group_split(
        X,
        y,
        groups,
        n_splits=4,
        random_state=RANDOM_STATE,
        fold_index=0,
    )

    # divides data into training+validation and test
    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    groups_trainval = groups[trainval_idx]

    meta_trainval = meta_df.iloc[trainval_idx].copy().reset_index(drop=True)
    meta_test = meta_df.iloc[test_idx].copy().reset_index(drop=True)

    # creates a stratified subject-level validation split
    train_idx_local, val_idx_local = _stratified_group_split(
        X_trainval,
        y_trainval,
        groups_trainval,
        n_splits=4,
        random_state=RANDOM_STATE + 1,
        fold_index=0,
    )

    # divides data in trainig, validation, test
    X_train, X_val = X_trainval[train_idx_local], X_trainval[val_idx_local]
    y_train, y_val = y_trainval[train_idx_local], y_trainval[val_idx_local]

    meta_val = meta_trainval.iloc[val_idx_local].copy().reset_index(drop=True)

    # checks if train and validation contain both classes
    if len(np.unique(y_train)) < 2:
        return None, None, None, None, None, "Training split contains only one class. Try a different split or more data."

    if len(np.unique(y_val)) < 2:
        return None, None, None, None, None, "Validation split contains only one class. Try a different split or more data."

    # normalizes sets using training stats
    X_train, X_val, X_test = _normalize_by_train(X_train, X_val, X_test)

    # chooses where the model will run
    device = _torch_device()

    # reproducibility
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    # creates cnn model
    model = SpectrogramCNN(
        in_channels=X_train.shape[1], # nr of eeg channels used as spectrogram channels
        dropout=CNN_DROPOUT, # droupout value
    ).to(device) # moves it to chosen device

    # for class imbalance
    pos_count = max(1, int((y_train == 1).sum()))
    neg_count = max(1, int((y_train == 0).sum()))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    # loss used for binary classification
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # optimizer to update model weights during training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CNN_LR,
        weight_decay=CNN_WEIGHT_DECAY,
    )

    # preps mini batches for training
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=CNN_BATCH_SIZE,
        shuffle=True,
    )

    # validation data preped as tensors and moved to same device as model
    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device)

    # for early stopping
    best_state = None # best model so far
    best_val_loss = float("inf") # best validation loss
    patience_left = CNN_PATIENCE # nr of epochs left before stopping
    epochs_ran = 0

    for epoch_idx in range(CNN_EPOCHS):
        epochs_ran = epoch_idx + 1
        model.train() # enables training behaviour
        for xb, yb in train_loader:
            xb = xb.to(device) # moves batch to device
            yb = yb.to(device) # moves batch to device

            if xb.shape[0] > 1:
                xb = xb + torch.randn_like(xb) * 0.015 # light noise augmentation to reduce overfitting

            optimizer.zero_grad() # clears old gradients
            logits = model(xb) # runs model forward
            loss = criterion(logits, yb) # computes loss
            loss.backward() # computes how each weight affected the error
            optimizer.step() # updates weights

        model.eval() # disables training behaviour
        with torch.no_grad(): # computes validation loss wo updating weights
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y).item())

        if val_loss < best_val_loss: # if validation loss improves
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict()) # saves model state
            patience_left = CNN_PATIENCE # reset patience
        else:
            patience_left -= 1 # reduce patience
            if patience_left <= 0: # stops training when patience reaches 0
                break

    if best_state is not None: # after training ends
        model.load_state_dict(best_state) # best model is restored

    # chooses the decision threshold from subject-level validation probabilities instead of fixed 0.5
    val_proba = _predict_proba(model, X_val, device)

    val_subject_df = meta_val[["subject_key", "label"]].copy()
    val_subject_df["proba_pd"] = val_proba.astype(float)
    val_subject_pred_df = subject_aggregate(val_subject_df)

    y_val_subject = val_subject_pred_df["label"].astype(int).values
    proba_val_subject = val_subject_pred_df["proba_pd"].astype(float).values

    threshold = safe_threshold(y_val_subject, proba_val_subject)

    val_subject_auc = None
    try:
        val_subject_auc = float(roc_auc_score(y_val_subject, proba_val_subject))
    except Exception:
        pass

    # prediction on test set
    proba = _predict_proba(model, X_test, device)
    pred = (proba >= threshold).astype(int) # converts probabilities into final class label

    # evaluation metrics
    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    cm = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
    extra_binary = _binary_extra_metrics(y_test, pred)

    # auc and roc curve
    auc = None
    roc = None
    try:
        auc = float(roc_auc_score(y_test, proba))
        fpr, tpr, thr = roc_curve(y_test, proba)
        roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": thr.tolist()}
    except Exception:
        pass

    # test prediction metadata
    test_meta = meta_test.copy().reset_index(drop=True) # original metadata
    test_meta["true_label"] = y_test.astype(int) # true label
    test_meta["pred_label"] = pred.astype(int) # predicted label
    test_meta["proba_pd"] = proba.astype(float) # PD probability
    test_meta["proba_hc"] = (1.0 - proba).astype(float) # HC probability
    test_meta["threshold"] = float(threshold) # threshold selected from validation set

    # computes subject level metrics by averaging window probabilities per subject
    subject_df = test_meta[["subject_key", "label", "proba_pd"]].copy()
    subject_pred_df = subject_aggregate(subject_df)
    y_subject = subject_pred_df["label"].astype(int).values
    proba_subject = subject_pred_df["proba_pd"].astype(float).values
    pred_subject = (proba_subject >= threshold).astype(int)

    subject_acc = float(accuracy_score(y_subject, pred_subject)) if len(y_subject) else None
    subject_f1 = float(f1_score(y_subject, pred_subject, zero_division=0)) if len(y_subject) else None
    subject_cm = confusion_matrix(y_subject, pred_subject, labels=[0, 1]).tolist() if len(y_subject) else None
    subject_extra_binary = _binary_extra_metrics(y_subject, pred_subject) if len(y_subject) else {}

    subject_auc = None
    try:
        subject_auc = float(roc_auc_score(y_subject, proba_subject))
    except Exception:
        pass

    prediction_counts = pd.Series(pred).value_counts().to_dict()
    subject_prediction_counts = pd.Series(pred_subject).value_counts().to_dict()

    train_global_idx = trainval_idx[train_idx_local]
    val_global_idx = trainval_idx[val_idx_local]
    test_global_idx = test_idx

    leakage_overlaps, split_subjects = _check_subject_leakage(
        groups,
        {
            "train": train_global_idx,
            "val": val_global_idx,
            "test": test_global_idx,
        }
    )

    if leakage_overlaps:
        messages = []
        for item in leakage_overlaps:
            messages.append(
                f"{item['split_a']} vs {item['split_b']}: {', '.join(item['subjects'])}"
            )

        return None, None, None, None, None, (
            "Subject leakage detected between splits: " + " | ".join(messages)
        )

    split_metrics = {}
    split_metrics.update(_split_info(y, groups, train_global_idx, "train"))
    split_metrics.update(_split_info(y, groups, val_global_idx, "val"))
    split_metrics.update(_split_info(y, groups, test_global_idx, "test"))
    split_metrics.update(_subject_list_metrics(split_subjects))
    split_metrics["subject_leakage_detected"] = False

    window_diagnostics = _probability_diagnostics(y_test, proba, "window")
    subject_diagnostics = _probability_diagnostics(y_subject, proba_subject, "subject")

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "balanced_accuracy": extra_binary["balanced_accuracy"],
        "sensitivity": extra_binary["sensitivity"],
        "specificity": extra_binary["specificity"],
        "subject_accuracy": subject_acc,
        "subject_f1": subject_f1,
        "subject_auc": subject_auc,
        "subject_balanced_accuracy": subject_extra_binary.get("balanced_accuracy"),
        "subject_sensitivity": subject_extra_binary.get("sensitivity"),
        "subject_specificity": subject_extra_binary.get("specificity"),
        "subject_confusion_matrix": subject_cm,
        "val_subject_auc": val_subject_auc,
        "threshold": float(threshold),
        "epochs_ran": int(epochs_ran),
        "best_val_loss": float(best_val_loss),
        "prediction_hc_windows": int(prediction_counts.get(0, 0)),
        "prediction_pd_windows": int(prediction_counts.get(1, 0)),
        "prediction_hc_subjects": int(subject_prediction_counts.get(0, 0)),
        "prediction_pd_subjects": int(subject_prediction_counts.get(1, 0)),
        "n_windows_total": int(summary["n_windows"]),
        "n_windows_train": int(len(X_train)),
        "n_windows_val": int(len(X_val)),
        "n_windows_test": int(len(X_test)),
        "n_subjects": int(summary["n_subjects"]),
        "n_recordings": int(summary["n_recordings"]),
        "n_channels": int(summary["n_channels"]),
        "cnn_input_channels": int(X.shape[1]),
        "window_samples": int(summary["window_samples"]),
        "sampling_rate": float(summary["sampling_rate"]) if summary["sampling_rate"] is not None else None,
        "visual_height": int(summary.get("visual_height", visual_height)),
        "visual_width": int(summary.get("visual_width", visual_width)),
        "spectrogram_channel_mode": str(summary.get("spectrogram_channel_mode", spectrogram_config["spectrogram_channel_mode"])),
        "input_type": "spectrogram",
        "model": "CompactSpectrogramCNN",
    }

    metrics.update(split_metrics)
    metrics.update(window_diagnostics)
    metrics.update(subject_diagnostics)

    return metrics, cm, roc, model, test_meta, None