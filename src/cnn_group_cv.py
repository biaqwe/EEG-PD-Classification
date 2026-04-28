import copy

import numpy as np
import pandas as pd

from src.config import (
    N_SPLITS,
    RANDOM_STATE,
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
    SPECTROGRAM_NPERSEG,
    SPECTROGRAM_OVERLAP_RATIO,
    SPECTROGRAM_CHANNEL_MODE,
)
from src.raw_eeg import load_brainvision_spectrograms
from src.ml_models import (
    SKLEARN_OK,
    TORCH_OK,
    SpectrogramCNN,
    _normalize_by_train,
    _predict_proba,
    subject_aggregate,
    safe_threshold,
    metrics_from_proba,
    _binary_extra_metrics,
    _probability_diagnostics,
    _subjects_with_multiple_labels,
    _check_subject_leakage,
    _split_info,
)

if SKLEARN_OK:
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score

if TORCH_OK:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset


def build_spectrogram_config(config: dict | None = None): # builds the config for spectrogram generation using defaults
    cfg = config or {}
    return {
        "window_sec": float(cfg.get("window_sec", RAW_WINDOW_SEC)),
        "step_sec": float(cfg.get("step_sec", RAW_STEP_SEC)),
        "max_windows_per_recording": int(cfg.get("max_windows_per_recording", RAW_MAX_WINDOWS_PER_RECORDING)),
        "use_bandpass": bool(cfg.get("use_bandpass", RAW_USE_BANDPASS)),
        "bandpass_low": float(cfg.get("bandpass_low", RAW_L_FREQ)),
        "bandpass_high": float(cfg.get("bandpass_high", RAW_H_FREQ)),
        "use_notch": bool(cfg.get("use_notch", RAW_USE_NOTCH)),
        "notch_freq": float(cfg.get("notch_freq", RAW_NOTCH_FREQ)),
        "visual_height": int(cfg.get("visual_height", SPECTROGRAM_HEIGHT)),
        "visual_width": int(cfg.get("visual_width", SPECTROGRAM_WIDTH)),
        "spectrogram_channel_mode": str(cfg.get("spectrogram_channel_mode", SPECTROGRAM_CHANNEL_MODE)),
        "spectrogram_nperseg": int(cfg.get("spectrogram_nperseg", SPECTROGRAM_NPERSEG)),
        "spectrogram_overlap_ratio": float(cfg.get("spectrogram_overlap_ratio", SPECTROGRAM_OVERLAP_RATIO)),
    }


def valid_inner_train_val_split(X, y, groups, n_splits=4, random_state=RANDOM_STATE, fold_index=0): # creates a valid train/val split
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups).astype(str)

    subject_df = pd.DataFrame({
        "subject_key": groups,
        "label": y,
    }).groupby("subject_key", as_index=False).agg(
        label=("label", "first")
    )

    min_groups_per_class = int(subject_df["label"].value_counts().min()) # finds the class with the least subjects and uses that nr as the max nr of splits to make sure each split has both classes represented
    effective_splits = min(int(n_splits), min_groups_per_class)
    candidate_splits = []

    if effective_splits >= 2:
        splitter = StratifiedGroupKFold( # creates stratified group k-fold splitter
            n_splits=effective_splits,
            shuffle=True,
            random_state=random_state,
        )
        candidate_splits = list(splitter.split(X, y, groups=groups))
        if candidate_splits: # rotates candidate splits based on fold index
            start = int(fold_index) % len(candidate_splits)
            candidate_splits = candidate_splits[start:] + candidate_splits[:start]

    for train_idx, val_idx in candidate_splits:
        if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[val_idx])) == 2:  # checks if each candidate split has both classes in train and val
            return train_idx, val_idx, None # returns the first valid one

    # tries random group shuffle splits with different seeds
    for extra_seed in range(20):
        try:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=RAW_VAL_SIZE,
                random_state=random_state + extra_seed,
            )
            train_idx, val_idx = next(splitter.split(X, y, groups=groups))
            if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[val_idx])) == 2:
                return train_idx, val_idx, None
        except Exception:
            pass

    return None, None, "Could not create a valid inner validation split with both HC and PD subjects."


def train_spectrogram_cnn_fold(X_train, y_train, X_val, y_val, seed): # trains a spectrogram cnn for one fold of the group cv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # sets device to gpu if available

    torch.manual_seed(seed) # sets random seeds for reproducibility
    np.random.seed(seed)
    if torch.cuda.is_available(): # sets random seed for all gpu devices
        torch.cuda.manual_seed_all(seed)

    model = SpectrogramCNN( # creates the cnn model using the config and input shape
        in_channels=X_train.shape[1], # nr of input channels
        dropout=CNN_DROPOUT, # dropout rate
    ).to(device)

    pos_count = max(1, int((y_train == 1).sum())) # positive class weight
    neg_count = max(1, int((y_train == 0).sum())) # negative class weight
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW( # optimizer for training
        model.parameters(),
        lr=CNN_LR,
        weight_decay=CNN_WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( #reduces lr if val loss doesnt improve for a few epochs
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    generator = torch.Generator() # sets random generator for data loading
    generator.manual_seed(seed) # sets random seed for data loading shuffling

    train_loader = DataLoader( # creates data loader for training data
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=CNN_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    val_x = torch.tensor(X_val, dtype=torch.float32).to(device) # moves validation data to device
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device) # moves validation labels to device

    # variables for early stopping
    best_state = None
    best_val_auc = -1.0
    best_val_loss = float("inf")
    patience_left = CNN_PATIENCE
    epochs_ran = 0

    for epoch_idx in range(CNN_EPOCHS): # runs the training loop for a max nr of epochs
        epochs_ran = epoch_idx + 1
        model.train() # sets model to training mode

        for xb, yb in train_loader: # iterates over training batches
            xb = xb.to(device)
            yb = yb.to(device)

            if xb.shape[0] > 1:
                xb = xb + torch.randn_like(xb) * 0.01

            optimizer.zero_grad() # zeroes the gradients before backprop
            logits = model(xb) # forward pass
            loss = criterion(logits, yb) # computes loss
            loss.backward() # backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0) # gradient clipping to prevent exploding gradients
            optimizer.step() # optimizer step to update weights

        model.eval() # sets model to evaluation mode for validation
        with torch.no_grad(): # disables gradient computation for validation
            val_logits = model(val_x) # forward pass on validation data
            val_loss = float(criterion(val_logits, val_y).item()) # computes validation loss
            val_proba = torch.sigmoid(val_logits).detach().cpu().numpy() # computes predicted probabilities for validation data

        scheduler.step(val_loss)

        val_auc = 0.5
        try:
            if len(np.unique(y_val)) == 2:
                val_auc = float(roc_auc_score(y_val, val_proba))
        except Exception:
            val_auc = 0.5

         # considers it an improvement if val_auc is better or if val_auc is the same but val_loss is better
        improved = val_auc > best_val_auc or (
            abs(val_auc - best_val_auc) < 1e-6 and val_loss < best_val_loss
        )

        if improved:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = CNN_PATIENCE
        else: # decreases patience if no improvement
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, device, best_val_loss, epochs_ran


def mean_std(values): # helper for mean and std
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None, None
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def add_mean_std(metrics_dict, prefix, values): # helper to add mean and std to metrics dict
    mean, std = mean_std(values)
    metrics_dict[f"{prefix}_mean"] = mean
    metrics_dict[f"{prefix}_std"] = std


def train_raw_eeg_cnn_group_cv(payloads: dict, config: dict | None = None, n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE): # trains and evaluates cnn group cv model
    if not SKLEARN_OK:
        return None, None, None, None, "scikit-learn not available in this environment."

    if not TORCH_OK:
        return None, None, None, None, "PyTorch is not available in this environment."

    spectrogram_config = build_spectrogram_config(config) # builds spectrogram config

    # turns raw eeg files into spectrogram images for model
    X, y, groups, meta_df, summary, err = load_brainvision_spectrograms(
    # X: spectrogram images, y: window labels, groups: subject or group ids, meta_df: window metadata, summary: info about ddataset, err: error msg
        payloads=payloads,
        config=spectrogram_config,
    )

    if err:
        return None, None, None, None, err

    if len(np.unique(y)) < 2: # cnn can only learn if both PD and HC are present so checks if data includes them both
        return None, None, None, None, "Need both classes (HC and PD) in the uploaded recordings."

    bad_subjects = _subjects_with_multiple_labels(y, groups) #
    if bad_subjects:
        return None, None, None, None, (
            "Subject label inconsistency detected. These subjects have both HC and PD labels: "
            + ", ".join(bad_subjects)
        )

    subject_df = pd.DataFrame({
        "subject_key": np.asarray(groups).astype(str),
        "label": np.asarray(y).astype(int),
    }).groupby("subject_key", as_index=False).agg(
        label=("label", "first")
    )

    min_subjects_per_class = int(subject_df["label"].value_counts().min())
    effective_splits = min(int(n_splits), min_subjects_per_class)

    if effective_splits < 2:
        return None, None, None, None, "Need at least 2 HC subjects and 2 PD subjects for subject-level CNN cross-validation."

    sgkf = StratifiedGroupKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )

    # window level metrics
    window_acc_values = []
    window_f1_values = []
    window_auc_values = []
    window_balanced_acc_values = []
    window_sensitivity_values = []
    window_specificity_values = []
    # subject level metrics
    subject_acc_values = []
    subject_f1_values = []
    subject_auc_values = []
    subject_balanced_acc_values = []
    subject_sensitivity_values = []
    subject_specificity_values = []
    # fold level info
    thresholds = []
    epochs_values = []
    val_losses = []
    val_subject_auc_values = []
    # confusion matrices
    window_cms = []
    subject_cms = []
    fold_rows = []
    sample_prediction_rows = []
    subject_prediction_rows = []
    # for probability diagnostics
    all_window_y = []
    all_window_proba = []
    all_subject_y = []
    all_subject_proba = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        trainval_subjects = set(np.asarray(groups)[trainval_idx].astype(str)) # gets the subjects in the trainval split
        test_subjects = set(np.asarray(groups)[test_idx].astype(str)) # gets the subjects in the test split
        overlap = sorted(trainval_subjects.intersection(test_subjects))
        if overlap: # checks for subject leakage
            return None, None, None, None, f"Subject leakage detected in CNN fold {fold_idx}: {', '.join(overlap)}"

        X_trainval = X[trainval_idx] # gets the data for the trainval split
        y_trainval = y[trainval_idx] # gets the labels for the trainval split
        groups_trainval = groups[trainval_idx] # gets the group ids for the trainval split

        train_idx_local, val_idx_local, split_err = valid_inner_train_val_split( # creates a trainval split within the trainval split
            X_trainval,
            y_trainval,
            groups_trainval,
            n_splits=4,
            random_state=random_state + fold_idx,
            fold_index=fold_idx - 1,
        )

        if split_err:
            return None, None, None, None, f"Fold {fold_idx}: {split_err}"

        train_idx = trainval_idx[train_idx_local] # gets train indices for current fold
        val_idx = trainval_idx[val_idx_local] # gets val indices for current fold

        leakage_overlaps, split_subjects = _check_subject_leakage( # checks for subject leakage
            groups,
            {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            }
        )

        if leakage_overlaps:
            messages = []
            for item in leakage_overlaps:
                messages.append(
                    f"{item['split_a']} vs {item['split_b']}: {', '.join(item['subjects'])}"
                )
            return None, None, None, None, f"Fold {fold_idx} subject leakage detected: " + " | ".join(messages)

        X_train = X[train_idx] # train data for curr fold
        y_train = y[train_idx]  # train labels for curr fold
        X_val = X[val_idx] # val data for curr fold
        y_val = y[val_idx] # val labels for curr fold
        X_test = X[test_idx] # test data for curr fold
        y_test = y[test_idx] # test labels for curr fold

        if len(np.unique(y_train)) < 2: # checks if train split has both classes
            return None, None, None, None, f"Fold {fold_idx}: training split contains only one class."

        if len(np.unique(y_val)) < 2: # checks if val split has both classes
            return None, None, None, None, f"Fold {fold_idx}: validation split contains only one class."

        if len(np.unique(y_test)) < 2: # checks if test split has both classes
            return None, None, None, None, f"Fold {fold_idx}: test split contains only one class."

        X_train, X_val, X_test = _normalize_by_train(X_train, X_val, X_test) # normalizes data based on train split

        model, device, best_val_loss, epochs_ran = train_spectrogram_cnn_fold( # trains the cnn for current fold
            X_train,
            y_train,
            X_val,
            y_val,
            seed=random_state + fold_idx, # uses different seed for each fold to get different data shuffling and augmentation patterns
        )

        val_proba = _predict_proba(model, X_val, device) # predicts probabilities for val split using the trained model
        meta_val = meta_df.iloc[val_idx].copy().reset_index(drop=True) # metadata for the val split
        val_subject_df = meta_val[["subject_key", "label"]].copy()  # creates dataframe with subject keys and labels for val split
        val_subject_df["proba_pd"] = val_proba.astype(float) # adds predicted probabilities to dataframe
        val_subject_pred_df = subject_aggregate(val_subject_df) # combines window level predictions into subject level for val split

        y_val_subject = val_subject_pred_df["label"].astype(int).values # true subject labels for val split
        proba_val_subject = val_subject_pred_df["proba_pd"].astype(float).values # predicted subject probabilities for val split
        threshold = safe_threshold(y_val_subject, proba_val_subject) # threshold for classification based on predicted probabilities

        val_subject_auc = float("nan")
        # computes validation subject level auc
        try:
            val_subject_auc = float(roc_auc_score(y_val_subject, proba_val_subject))
        except Exception:
            pass

        proba_test = _predict_proba(model, X_test, device) # predicts probabilities for test split using the trained model
        pred_test = (proba_test >= threshold).astype(int) # predicted labels for test split based on threshold

        acc_w, f1_w, auc_w, cm_w = metrics_from_proba(y_test, proba_test, threshold) # computes window level metrics for test split
        extra_w = _binary_extra_metrics(y_test, pred_test) # computes extra window level metrics

        test_meta = meta_df.iloc[test_idx].copy().reset_index(drop=True) # metadata for test split
        test_meta["true_label"] = y_test.astype(int)
        test_meta["pred_label"] = pred_test.astype(int)
        test_meta["proba_pd"] = proba_test.astype(float)
        test_meta["proba_hc"] = (1.0 - proba_test).astype(float)
        test_meta["threshold"] = float(threshold)
        test_meta["fold"] = int(fold_idx)

        subject_df_fold = test_meta[["subject_key", "label", "proba_pd"]].copy()
        subject_pred_df = subject_aggregate(subject_df_fold)
        y_subject = subject_pred_df["label"].astype(int).values
        proba_subject = subject_pred_df["proba_pd"].astype(float).values
        pred_subject = (proba_subject >= threshold).astype(int)

        # computes subject level metrics for test split
        acc_s, f1_s, auc_s, cm_s = metrics_from_proba(y_subject, proba_subject, threshold)
        extra_s = _binary_extra_metrics(y_subject, pred_subject)

        window_acc_values.append(acc_w)
        window_f1_values.append(f1_w)
        window_auc_values.append(auc_w)
        window_balanced_acc_values.append(extra_w["balanced_accuracy"])
        window_sensitivity_values.append(extra_w["sensitivity"])
        window_specificity_values.append(extra_w["specificity"])

        subject_acc_values.append(acc_s)
        subject_f1_values.append(f1_s)
        subject_auc_values.append(auc_s)
        subject_balanced_acc_values.append(extra_s["balanced_accuracy"])
        subject_sensitivity_values.append(extra_s["sensitivity"])
        subject_specificity_values.append(extra_s["specificity"])

        thresholds.append(float(threshold))
        epochs_values.append(int(epochs_ran))
        val_losses.append(float(best_val_loss))
        val_subject_auc_values.append(val_subject_auc)

        window_cms.append(cm_w)
        subject_cms.append(cm_s)

        all_window_y.extend(y_test.astype(int).tolist())
        all_window_proba.extend(proba_test.astype(float).tolist())
        all_subject_y.extend(y_subject.astype(int).tolist())
        all_subject_proba.extend(proba_subject.astype(float).tolist())

        split_metrics = {}
        split_metrics.update(_split_info(y, groups, train_idx, "train"))
        split_metrics.update(_split_info(y, groups, val_idx, "val"))
        split_metrics.update(_split_info(y, groups, test_idx, "test"))

        fold_rows.append({ # info about the current fold
            "fold": int(fold_idx),
            "train_subjects": split_metrics.get("train_subjects"),
            "val_subjects": split_metrics.get("val_subjects"),
            "test_subjects": split_metrics.get("test_subjects"),
            "train_hc_subjects": split_metrics.get("train_hc_subjects"),
            "train_pd_subjects": split_metrics.get("train_pd_subjects"),
            "val_hc_subjects": split_metrics.get("val_hc_subjects"),
            "val_pd_subjects": split_metrics.get("val_pd_subjects"),
            "test_hc_subjects": split_metrics.get("test_hc_subjects"),
            "test_pd_subjects": split_metrics.get("test_pd_subjects"),
            "threshold": float(threshold),
            "epochs_ran": int(epochs_ran),
            "best_val_loss": float(best_val_loss),
            "val_subject_auc": None if np.isnan(val_subject_auc) else float(val_subject_auc),
            "window_acc": float(acc_w),
            "window_f1": float(f1_w),
            "window_auc": float(auc_w),
            "window_balanced_accuracy": float(extra_w["balanced_accuracy"]),
            "window_sensitivity": float(extra_w["sensitivity"]),
            "window_specificity": float(extra_w["specificity"]),
            "subject_acc": float(acc_s),
            "subject_f1": float(f1_s),
            "subject_auc": float(auc_s),
            "subject_balanced_accuracy": float(extra_s["balanced_accuracy"]),
            "subject_sensitivity": float(extra_s["sensitivity"]),
            "subject_specificity": float(extra_s["specificity"]),
            "train_subject_keys": ", ".join(sorted(split_subjects["train"])),
            "val_subject_keys": ", ".join(sorted(split_subjects["val"])),
            "test_subject_keys": ", ".join(sorted(split_subjects["test"])),
        })

        for local_i in range(len(test_meta)): # saves test window level predictions for current fold
            row = test_meta.iloc[local_i]
            start_sample = int(row.get("start_sample", -1))
            sample_prediction_rows.append({
                "row_index": int(test_idx[local_i]),
                "fold": int(fold_idx),
                "recording": str(row.get("recording", "")),
                "subject_key": str(row.get("subject_key", "")),
                "subject_id": str(row.get("subject_id", row.get("subject_key", ""))),
                "window_index": int(row.get("window_index", -1)),
                "window_start": start_sample,
                "start_sample": start_sample,
                "end_sample": int(row.get("end_sample", -1)),
                "true_label": int(row["true_label"]),
                "pred_label": int(row["pred_label"]),
                "proba_pd": float(row["proba_pd"]),
                "proba_hc": float(row["proba_hc"]),
                "threshold": float(threshold),
            })

        for local_i in range(len(subject_pred_df)): # saves test subject level predictions for current fold
            row = subject_pred_df.iloc[local_i]
            subject_prediction_rows.append({
                "fold": int(fold_idx),
                "subject_key": str(row.get("subject_key", "")),
                "subject_id": str(row.get("subject_id", row.get("subject_key", ""))),
                "true_label": int(row["label"]),
                "pred_label": int(pred_subject[local_i]),
                "proba_pd": float(row["proba_pd"]),
                "proba_hc": float(1.0 - row["proba_pd"]),
                "threshold": float(threshold),
            })

    # compiles all the metrics and results into a dict
    metrics = {}
    add_mean_std(metrics, "window_acc", window_acc_values)
    add_mean_std(metrics, "window_f1", window_f1_values)
    add_mean_std(metrics, "window_auc", window_auc_values)
    add_mean_std(metrics, "window_balanced_accuracy", window_balanced_acc_values)
    add_mean_std(metrics, "window_sensitivity", window_sensitivity_values)
    add_mean_std(metrics, "window_specificity", window_specificity_values)

    add_mean_std(metrics, "subject_acc", subject_acc_values)
    add_mean_std(metrics, "subject_f1", subject_f1_values)
    add_mean_std(metrics, "subject_auc", subject_auc_values)
    add_mean_std(metrics, "subject_balanced_accuracy", subject_balanced_acc_values)
    add_mean_std(metrics, "subject_sensitivity", subject_sensitivity_values)
    add_mean_std(metrics, "subject_specificity", subject_specificity_values)

    add_mean_std(metrics, "threshold", thresholds)
    add_mean_std(metrics, "epochs_ran", epochs_values)
    add_mean_std(metrics, "best_val_loss", val_losses)
    add_mean_std(metrics, "val_subject_auc", val_subject_auc_values)

    window_cm_sum = np.sum(np.stack(window_cms, axis=0), axis=0).astype(int).tolist()
    subject_cm_sum = np.sum(np.stack(subject_cms, axis=0), axis=0).astype(int).tolist()

    window_diagnostics = _probability_diagnostics(all_window_y, all_window_proba, "window_cv")
    subject_diagnostics = _probability_diagnostics(all_subject_y, all_subject_proba, "subject_cv")

    sample_predictions_df = pd.DataFrame(sample_prediction_rows).sort_values(["fold", "row_index"]).reset_index(drop=True)
    subject_predictions_df = pd.DataFrame(subject_prediction_rows).sort_values(["fold", "subject_key"]).reset_index(drop=True)

    metrics.update({
        "n_splits": int(effective_splits),
        "requested_n_splits": int(n_splits),
        "subjects": int(subject_df["subject_key"].nunique()),
        "n_subjects": int(subject_df["subject_key"].nunique()),
        "hc_subjects": int((subject_df["label"] == 0).sum()),
        "pd_subjects": int((subject_df["label"] == 1).sum()),
        "n_windows_total": int(summary["n_windows"]),
        "n_recordings": int(summary["n_recordings"]),
        "n_channels": int(summary["n_channels"]),
        "cnn_input_channels": int(X.shape[1]),
        "window_samples": int(summary["window_samples"]),
        "sampling_rate": float(summary["sampling_rate"]) if summary["sampling_rate"] is not None else None,
        "visual_height": int(summary.get("visual_height", spectrogram_config["visual_height"])),
        "visual_width": int(summary.get("visual_width", spectrogram_config["visual_width"])),
        "spectrogram_channel_mode": str(summary.get("spectrogram_channel_mode", spectrogram_config["spectrogram_channel_mode"])),
        "spectrogram_nperseg": int(summary.get("spectrogram_nperseg", spectrogram_config["spectrogram_nperseg"])),
        "spectrogram_overlap_ratio": float(summary.get("spectrogram_overlap_ratio", spectrogram_config["spectrogram_overlap_ratio"])),
        "input_type": "spectrogram",
        "model": "CompactSpectrogramCNN",
        "validation_strategy": "5-fold subject-level cross-validation",
        "subject_leakage_detected": False,
        "fold_details": fold_rows,
        "subject_predictions": subject_predictions_df.to_dict(orient="records"),
        "preprocessing_config": spectrogram_config,
    })

    metrics.update(window_diagnostics)
    metrics.update(subject_diagnostics)

    return metrics, window_cm_sum, subject_cm_sum, sample_predictions_df, None