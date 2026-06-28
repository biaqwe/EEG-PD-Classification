import copy

import numpy as np
import pandas as pd

from src.config import (
    RANDOM_STATE,
    RAW_WINDOW_SEC,
    RAW_STEP_SEC,
    RAW_MAX_WINDOWS_PER_RECORDING,
    RAW_L_FREQ,
    RAW_H_FREQ,
    RAW_NOTCH_FREQ,
    RAW_USE_BANDPASS,
    RAW_USE_NOTCH,
    SPECTROGRAM_HEIGHT,
    SPECTROGRAM_WIDTH,
)
from src.raw_eeg import load_brainvision_spectrograms
from src.ml_models import (
    SKLEARN_OK,
    TORCH_OK,
    _binary_extra_metrics,
    _check_subject_leakage,
    _normalize_by_train,
    _predict_proba,
    _probability_diagnostics,
    _split_info,
    _stratified_group_split,
    _subject_list_metrics,
    _subjects_with_multiple_labels,
    safe_threshold,
    subject_aggregate,
)

if SKLEARN_OK:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

if TORCH_OK:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset


# These settings are used only by the normal Train CNN button.
# CNN Group CV is not affected because it still imports SpectrogramCNN from src.ml_models.
SIMPLE_CNN_EPOCHS = 60
SIMPLE_CNN_BATCH_SIZE = 16
SIMPLE_CNN_LR = 3e-4
SIMPLE_CNN_WEIGHT_DECAY = 5e-4
SIMPLE_CNN_DROPOUT = 0.45
SIMPLE_CNN_PATIENCE = 10


def _simple_cnn_device():
    # Old Train CNN behavior: CUDA if available, otherwise CPU.
    # This avoids Apple MPS for the simple CNN, because the older high-scoring run did not use MPS.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SimpleSpectrogramCNN(nn.Module):
    # Compact CNN used only by Train CNN.
    def __init__(self, in_channels: int, dropout: float = SIMPLE_CNN_DROPOUT):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),

            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


def train_raw_eeg_simple_cnn(payloads: dict, config: dict | None = None):
    # Train/evaluate the legacy compact CNN used by the normal Train CNN button.
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

    visual_height = int(cfg.get("visual_height", SPECTROGRAM_HEIGHT))
    visual_width = int(cfg.get("visual_width", SPECTROGRAM_WIDTH))

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

    X, y, groups, meta_df, summary, err = load_brainvision_spectrograms(
        payloads=payloads,
        config=spectrogram_config,
    )
    if err:
        return None, None, None, None, None, err

    if len(np.unique(y)) < 2:
        return None, None, None, None, None, "Need both classes (HC and PD) in the uploaded recordings."

    bad_subjects = _subjects_with_multiple_labels(y, groups)
    if bad_subjects:
        return None, None, None, None, None, (
            "Subject label inconsistency detected. These subjects have both HC and PD labels: "
            + ", ".join(bad_subjects)
        )

    trainval_idx, test_idx = _stratified_group_split(
        X,
        y,
        groups,
        n_splits=4,
        random_state=RANDOM_STATE,
        fold_index=0,
    )

    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    groups_trainval = groups[trainval_idx]

    meta_trainval = meta_df.iloc[trainval_idx].copy().reset_index(drop=True)
    meta_test = meta_df.iloc[test_idx].copy().reset_index(drop=True)

    train_idx_local, val_idx_local = _stratified_group_split(
        X_trainval,
        y_trainval,
        groups_trainval,
        n_splits=4,
        random_state=RANDOM_STATE + 1,
        fold_index=0,
    )

    X_train, X_val = X_trainval[train_idx_local], X_trainval[val_idx_local]
    y_train, y_val = y_trainval[train_idx_local], y_trainval[val_idx_local]
    meta_val = meta_trainval.iloc[val_idx_local].copy().reset_index(drop=True)

    if len(np.unique(y_train)) < 2:
        return None, None, None, None, None, "Training split contains only one class. Try a different split or more data."

    if len(np.unique(y_val)) < 2:
        return None, None, None, None, None, "Validation split contains only one class. Try a different split or more data."

    X_train, X_val, X_test = _normalize_by_train(X_train, X_val, X_test)

    device = _simple_cnn_device()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    model = SimpleSpectrogramCNN(
        in_channels=X_train.shape[1],
        dropout=SIMPLE_CNN_DROPOUT,
    ).to(device)

    pos_count = max(1, int((y_train == 1).sum()))
    neg_count = max(1, int((y_train == 0).sum()))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SIMPLE_CNN_LR,
        weight_decay=SIMPLE_CNN_WEIGHT_DECAY,
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=SIMPLE_CNN_BATCH_SIZE,
        shuffle=True,
    )

    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_state = None
    best_val_loss = float("inf")
    patience_left = SIMPLE_CNN_PATIENCE
    epochs_ran = 0

    for epoch_idx in range(SIMPLE_CNN_EPOCHS):
        epochs_ran = epoch_idx + 1
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if xb.shape[0] > 1:
                xb = xb + torch.randn_like(xb) * 0.015

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
            patience_left = SIMPLE_CNN_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

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

    proba = _predict_proba(model, X_test, device)
    pred = (proba >= threshold).astype(int)

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    cm = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
    extra_binary = _binary_extra_metrics(y_test, pred)

    auc = None
    roc = None
    try:
        auc = float(roc_auc_score(y_test, proba))
        fpr, tpr, thr = roc_curve(y_test, proba)
        roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": thr.tolist()}
    except Exception:
        pass

    test_meta = meta_test.copy().reset_index(drop=True)
    test_meta["true_label"] = y_test.astype(int)
    test_meta["pred_label"] = pred.astype(int)
    test_meta["proba_pd"] = proba.astype(float)
    test_meta["proba_hc"] = (1.0 - proba).astype(float)
    test_meta["threshold"] = float(threshold)

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
        },
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
        "model": "SimpleSpectrogramCNN",
        "training_variant": "simple_cnn_legacy_compact",
        "cnn_epochs_configured": int(SIMPLE_CNN_EPOCHS),
        "cnn_lr": float(SIMPLE_CNN_LR),
        "cnn_weight_decay": float(SIMPLE_CNN_WEIGHT_DECAY),
        "cnn_dropout": float(SIMPLE_CNN_DROPOUT),
        "cnn_patience": int(SIMPLE_CNN_PATIENCE),
        "cnn_batch_size": int(SIMPLE_CNN_BATCH_SIZE),
        "device": str(device),
    }

    metrics.update(split_metrics)
    metrics.update(window_diagnostics)
    metrics.update(subject_diagnostics)

    return metrics, cm, roc, model, test_meta, None