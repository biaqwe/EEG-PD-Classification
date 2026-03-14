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


def safe_threshold(y_true, proba): # computes classification threshold based on roc curve
    fpr, tpr, thr = roc_curve(y_true, proba) # calculates roc curve
    # fpr: false positive rate, tpr: true positive rate, thr: probability thresholds
    j = tpr - fpr # computes j score for each threshold to measure how well it separates the two classes
    idx = int(np.argmax(j)) # finds index of max j score
    t = float(thr[idx]) # selects probability threshold of the best point
    if not np.isfinite(t):
        t = 0.5
    return t


def metrics_from_proba(y_true, proba, thr): # calculates metrics
    pred = (proba >= thr).astype(int) # converts probabilities intro predictied classes
    acc = float(accuracy_score(y_true, pred)) # accuracy
    f1 = float(f1_score(y_true, pred)) # f1 score
    auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan") # area under the roc curve
    cm = confusion_matrix(y_true, pred) # confusion matrix
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
        # computes best threshold for this fold from roc curve
        thr = safe_threshold(y_test, proba_test)
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


def _normalize_by_train(X_train, X_val, X_test): # normalizes training, validation and test data
    mean = X_train.mean(axis=(0, 2), keepdims=True) # avg value of training data per eeg channel
    std = X_train.std(axis=(0, 2), keepdims=True) # std of training data per eeg channel
    std[std < 1e-8] = 1.0 # avoids /0

    X_train = (X_train - mean) / std # z score normalization
    X_val = (X_val - mean) / std # same training avg and std to not leak info
    X_test = (X_test - mean) / std # same training stats
    return X_train, X_val, X_test


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

    # turns raw eeg files into data for model
    X, y, groups, meta_df, summary, err = load_brainvision_windows(
    # X: eeg windows, y: window labels, groups: subject or group ids, meta_df: window metadata, summary: info about ddataset, err: error msg
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

    # cnn can only learn if both PD and HC are present so checks if data includes them both
    if len(np.unique(y)) < 2:
        return None, None, None, None, None, "Need both classes (HC and PD) in the uploaded recordings."

    # creates the first split
    gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE) # GroupShuffleSplit so windows from one subject stay together
    trainval_idx, test_idx = next(gss_test.split(X, y, groups))

    # divides data into training+validation and test
    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    groups_trainval = groups[trainval_idx]

    # creates the second split
    val_fraction = RAW_VAL_SIZE
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=RANDOM_STATE)
    train_idx_local, val_idx_local = next(gss_val.split(X_trainval, y_trainval, groups_trainval))

    # divides data in trainig, validation, test
    X_train, X_val = X_trainval[train_idx_local], X_trainval[val_idx_local]
    y_train, y_val = y_trainval[train_idx_local], y_trainval[val_idx_local]

    # normalizes sets using training stats
    X_train, X_val, X_test = _normalize_by_train(X_train, X_val, X_test)

    # adds an extra dimension bcs cnn expects data in 4d format
    X_train = X_train[:, None, :, :]
    X_val = X_val[:, None, :, :]
    X_test = X_test[:, None, :, :]

    # chooses where the model will run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reproducibility
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # creates cnn model
    model = EEGNetLite(
        n_channels=X_train.shape[2], # nr of eeeg channels
        n_samples=X_train.shape[3], # nr of time samples
        dropout=CNN_DROPOUT, # droupout value
    ).to(device) # moves it to chosen device

    # for class imbalance
    pos_count = max(1, int((y_train == 1).sum()))
    neg_count = max(1, int((y_train == 0).sum()))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    # loss used for binary classification
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # optimizer to update model weights during training
    optimizer = torch.optim.Adam(
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

    for _ in range(CNN_EPOCHS):
        model.train() # enables training behaviour
        for xb, yb in train_loader:
            xb = xb.to(device) # moves batch to device
            yb = yb.to(device) # moves batch to device

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

    model.eval() # prediction on test set
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32).to(device) # converts test data to tensor
        logits = model(test_x) # runs model
        proba = torch.sigmoid(logits).cpu().numpy() # PD probabilities for test windows

    pred = (proba >= 0.5).astype(int) # converts probabilities into final class label

    # evaluation metrics
    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred))
    cm = confusion_matrix(y_test, pred).tolist()

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
    test_meta = meta_df.iloc[test_idx].copy().reset_index(drop=True) # original metadata
    test_meta["true_label"] = y_test.astype(int) # true label
    test_meta["pred_label"] = pred.astype(int) # predicted label
    test_meta["proba_pd"] = proba.astype(float) # PD probability
    test_meta["proba_hc"] = (1.0 - proba).astype(float) # HC probability

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