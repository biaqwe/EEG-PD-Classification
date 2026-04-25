import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
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
    SPECTROGRAM_NPERSEG,
    SPECTROGRAM_OVERLAP_RATIO,
    SPECTROGRAM_CHANNEL_MODE,
)

def infer_label_from_name(name: str): # tries to guess class label from filename
    s = name.lower()

    if "control" in s or "_hc" in s or "-hc" in s or s.startswith("hc"):
        return 0

    if "pd" in s or "parkinson" in s or "patient" in s:
        return 1

    return None


def subject_key_from_name(name: str): # extracts subject id from filename
    stem = Path(name).stem
    m = re.search(r"([A-Za-z]+)(\d+)", stem)
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"
    return stem.lower()


def build_brainvision_payload(uploaded_files): # processes uploaded raw eeg files
    if not uploaded_files:
        return None, None, "No files uploaded."

    payloads = {} # for file data
    grouped = {} # for files belonging to the same recording

    for f in uploaded_files:
        name = Path(f.name).name
        ext = Path(name).suffix.lower()

        if ext not in [".vhdr", ".eeg", ".vmrk"]:
            continue

        data = f.getvalue()
        payloads[name] = data

        stem = Path(name).stem
        grouped.setdefault(stem, set()).add(ext)

    rows = []
    valid_recordings = 0

    for stem, exts in sorted(grouped.items()):
        has_vhdr = ".vhdr" in exts
        has_eeg = ".eeg" in exts
        has_vmrk = ".vmrk" in exts
        complete = has_vhdr and has_eeg and has_vmrk # checks if recording is complete

        label = infer_label_from_name(stem)
        subject_key = subject_key_from_name(stem)

        if complete:
            valid_recordings += 1

        rows.append({ # recording info
            "recording": stem, # recording name
            "subject_key": subject_key, # subject if
            "label_guess": label, # guessed class from filename
            "has_vhdr": has_vhdr, # header present
            "has_eeg": has_eeg, # sig present
            "has_vmrk": has_vmrk, # marker present
            "complete_triplet": complete, # all files present
        })

    manifest_df = pd.DataFrame(rows) # rows to dataframe

    if manifest_df.empty:
        return None, None, "No valid BrainVision files found."

    if valid_recordings == 0:
        return None, None, "No complete BrainVision triplets found (.vhdr + .eeg + .vmrk)."

    return payloads, manifest_df, None


def _rewrite_brainvision_links(file_path: Path): # rewrites eeg and marker file refs
    ext = file_path.suffix.lower()
    stem = file_path.stem

    if ext not in [".vhdr", ".vmrk"]:
        return

    text = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    new_lines = []
    for line in lines:
        if line.startswith("DataFile="): # if a line defines eeg file
            new_lines.append(f"DataFile={stem}.eeg") # rewrites it so header points to correct eeg file
        elif line.startswith("MarkerFile="): # if a line defines marker file
            new_lines.append(f"MarkerFile={stem}.vmrk") # rewrites it so header points to correct marker file
        else:
            new_lines.append(line)

    file_path.write_text("\n".join(new_lines), encoding="utf-8")


def materialize_brainvision_payload(payloads: dict): # writes uploaded files to temp folder and fixes links
    # creates a temp folder on system
    tmpdir = tempfile.TemporaryDirectory()
    root = Path(tmpdir.name)

    # writes uploaded files in temp folder
    for filename, data in payloads.items():
        out_path = root / Path(filename).name
        out_path.write_bytes(data)

    # fixes links for all files in temp folder
    for file_path in root.iterdir():
        if file_path.suffix.lower() in [".vhdr", ".vmrk"]:
            _rewrite_brainvision_links(file_path)

    vhdr_paths = sorted(root.glob("*.vhdr"))
    return tmpdir, vhdr_paths


def _preprocessing_values(config: dict | None = None): # reads saved preprocessing settings or defaults
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
        "spectrogram_nperseg": int(cfg.get("spectrogram_nperseg", SPECTROGRAM_NPERSEG)),
        "spectrogram_overlap_ratio": float(cfg.get("spectrogram_overlap_ratio", SPECTROGRAM_OVERLAP_RATIO)),
        "spectrogram_channel_mode": str(cfg.get("spectrogram_channel_mode", SPECTROGRAM_CHANNEL_MODE)).lower(),
    }

def _apply_saved_preprocessing(raw, config: dict | None = None): # applies saved filtering config to one raw recording
    cfg = _preprocessing_values(config)

    if cfg["use_bandpass"]: # applies bandpass if enabled
        raw.filter(l_freq=cfg["bandpass_low"], h_freq=cfg["bandpass_high"], verbose="ERROR")

    if cfg["use_notch"] and cfg["notch_freq"] > 0: # applies notch if enabled
        raw.notch_filter(freqs=[cfg["notch_freq"]], verbose="ERROR")

    return raw


def _resize_2d(arr: np.ndarray, out_h: int, out_w: int): # resizes one spectrogram channel to fixed image size
    try:
        from scipy.ndimage import zoom
    except Exception:
        return None

    if arr.shape[0] == out_h and arr.shape[1] == out_w:
        return arr.astype(np.float32)

    zoom_h = out_h / max(1, arr.shape[0])
    zoom_w = out_w / max(1, arr.shape[1])
    resized = zoom(arr, (zoom_h, zoom_w), order=1)

    if resized.shape[0] != out_h or resized.shape[1] != out_w:
        fixed = np.zeros((out_h, out_w), dtype=np.float32)
        h = min(out_h, resized.shape[0])
        w = min(out_w, resized.shape[1])
        fixed[:h, :w] = resized[:h, :w]
        resized = fixed

    return resized.astype(np.float32)


def _window_to_spectrogram(window: np.ndarray, sfreq: float, config: dict | None = None): # converts one eeg window to a spectrogram image tensor
    try:
        from scipy.signal import spectrogram
    except Exception:
        return None

    cfg = _preprocessing_values(config)
    visual_height = int(cfg["visual_height"])
    visual_width = int(cfg["visual_width"])

    n_samples = int(window.shape[1])
    nperseg = min(int(cfg["spectrogram_nperseg"]), n_samples)
    nperseg = max(8, nperseg)
    noverlap = int(nperseg * float(cfg["spectrogram_overlap_ratio"]))
    noverlap = min(max(0, noverlap), nperseg - 1)

    freqs, _, pxx = spectrogram(
        window,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=1,
        scaling="density",
        mode="psd",
    )

    if cfg["use_bandpass"]: # keeps the same frequency range as preprocessing when possible
        freq_mask = (freqs >= cfg["bandpass_low"]) & (freqs <= cfg["bandpass_high"])
        if freq_mask.any():
            pxx = pxx[:, freq_mask, :]

    pxx = np.log10(pxx.astype(np.float32) + 1e-12) # log-power spectrogram
    if cfg["spectrogram_channel_mode"] == "mean": # averages all eeg channels into one spectrogram image
        pxx = np.mean(pxx, axis=0, keepdims=True)

    channels = []
    for ch_idx in range(pxx.shape[0]):
        resized = _resize_2d(pxx[ch_idx], visual_height, visual_width)
        if resized is None:
            return None
        channels.append(resized)

    return np.stack(channels, axis=0).astype(np.float32)


def load_brainvision_windows( # prepares raw eeg windows for cnn training
    payloads: dict,
    window_sec: float,
    step_sec: float,
    max_windows_per_recording: int,
    use_bandpass: bool,
    l_freq: float,
    h_freq: float,
    use_notch: bool,
    notch_freq: float,
):
    try:
        import mne
    except Exception as e:
        return None, None, None, None, None, f"MNE is not available: {e}"

    tmpdir, vhdr_paths = materialize_brainvision_payload(payloads)

    X_list = [] # eeg windows
    y_list = [] # window labels
    groups = [] # subject ids
    rows = [] # metadata rows
    channels_count = None # expected nr of channels
    sfreq_ref = None # sampling freq ref

    try: # makes sure temp folder is deleted at the end
        for vhdr_path in vhdr_paths:
            stem = vhdr_path.stem
            label = infer_label_from_name(stem)

            if label is None:
                continue

            subject_key = subject_key_from_name(stem)

            try: # loads eeg recording
                raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR") # preload means sig data is loaded into memory immediately
            except Exception as e:
                return None, None, None, None, None, f"Failed to read {vhdr_path.name}: {e}"

            raw.pick("eeg") # keeps only eeg sigs

            if len(raw.ch_names) == 0:
                continue

            if use_bandpass: # applies bandpass if enabled
                raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")

            if use_notch and notch_freq > 0: # applies notch if enabled
                raw.notch_filter(freqs=[notch_freq], verbose="ERROR")

            data = raw.get_data().astype(np.float32) # extracts eeg data as numpy array
            sfreq = float(raw.info["sfreq"]) # reads sampling freq

            if sfreq_ref is None:
                sfreq_ref = sfreq

            if channels_count is None:
                channels_count = data.shape[0]

            if data.shape[0] != channels_count:
                continue

            # converts window aand step from seconds to sampmles
            win_samples = int(window_sec * sfreq)
            step_samples = int(step_sec * sfreq)

            if win_samples <= 0 or step_samples <= 0:
                continue

            # calculates where each window should start in sig
            total = data.shape[1]
            starts = list(range(0, max(0, total - win_samples + 1), step_samples))

            # only max nr of windows per recording are kept
            if max_windows_per_recording is not None and len(starts) > max_windows_per_recording:
                starts = starts[:max_windows_per_recording]

            for i, start in enumerate(starts):
                # slices one eeg sig from all channels
                end = start + win_samples
                window = data[:, start:end]

                if window.shape[1] != win_samples:
                    continue

                X_list.append(window)
                y_list.append(label)
                groups.append(subject_key)

                # stores window metadata
                rows.append({
                    "recording": stem,
                    "subject_key": subject_key,
                    "label": int(label),
                    "window_index": int(i),
                    "start_sample": int(start),
                    "end_sample": int(end),
                })

    finally:
        tmpdir.cleanup() # deletes temp folder

    if not X_list:
        return None, None, None, None, None, "Could not generate windows from the uploaded BrainVision files."

    # builds arrays and metadata table
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(groups)
    meta_df = pd.DataFrame(rows)

    # dataset summary
    summary = {
        "n_windows": int(X.shape[0]),
        "n_recordings": int(meta_df["recording"].nunique()),
        "n_subjects": int(meta_df["subject_key"].nunique()),
        "n_channels": int(X.shape[1]),
        "window_samples": int(X.shape[2]),
        "sampling_rate": float(sfreq_ref) if sfreq_ref is not None else None,
        "class_counts": {
            "hc": int((y == 0).sum()),
            "pd": int((y == 1).sum()),
        },
    }

    return X, y, groups, meta_df, summary, None


def load_brainvision_spectrograms(payloads: dict, config: dict | None = None): # prepares spectrogram images for cnn training
    cfg = _preprocessing_values(config)

    # first loads preprocessed eeg windows using the same saved configuration
    X_windows, y, groups, meta_df, summary, err = load_brainvision_windows(
    # X_windows: eeg windows, y: window labels, groups: subject ids, meta_df: metadata, summary: data info, err: error msg
        payloads=payloads,
        window_sec=cfg["window_sec"],
        step_sec=cfg["step_sec"],
        max_windows_per_recording=cfg["max_windows_per_recording"],
        use_bandpass=cfg["use_bandpass"],
        l_freq=cfg["bandpass_low"],
        h_freq=cfg["bandpass_high"],
        use_notch=cfg["use_notch"],
        notch_freq=cfg["notch_freq"],
    )

    if err:
        return None, None, None, None, None, err

    sfreq = float(summary["sampling_rate"])
    spec_list = [] # spectrogram tensors

    for window in X_windows:
        spec = _window_to_spectrogram(window, sfreq=sfreq, config=cfg)
        if spec is None:
            return None, None, None, None, None, "Could not generate spectrograms. Make sure scipy is installed."
        spec_list.append(spec)

    X = np.stack(spec_list, axis=0).astype(np.float32) # shape: samples x channels x height x width

    summary = dict(summary)
    summary.update({
        "input_type": "spectrogram",
        "visual_height": int(cfg["visual_height"]),
        "visual_width": int(cfg["visual_width"]),
        "spectrogram_nperseg": int(cfg["spectrogram_nperseg"]),
        "spectrogram_overlap_ratio": float(cfg["spectrogram_overlap_ratio"]),
        "spectrogram_channel_mode": str(cfg["spectrogram_channel_mode"]),
        "cnn_input_channels": int(X.shape[1]),
    })

    return X, y, groups, meta_df, summary, None


def load_brainvision_recording_preview(payloads: dict, recording: str, seconds: float | None = None, config: dict | None = None): # loads one raw eeg recoring and prepares a preview for visualization
    try:
        import mne
    except Exception as e:
        return None, None, None, f"MNE is not available: {e}"

    tmpdir, vhdr_paths = materialize_brainvision_payload(payloads)

    try:
        # searches for the selected recording
        selected_path = None
        for vhdr_path in vhdr_paths:
            if vhdr_path.stem == recording:
                selected_path = vhdr_path
                break

        if selected_path is None:
            return None, None, None, f"Recording not found: {recording}"

        raw = mne.io.read_raw_brainvision(selected_path, preload=True, verbose="ERROR") # reads the file using mne
        # preload=True: loads the sig data directly into memory, verbose="ERROR": suppresses extra nonerror mne msgs
        raw.pick("eeg")

        if len(raw.ch_names) == 0:
            return None, None, None, "No EEG channels found in the selected recording."

        # applies the same saved filtering settings used by the cnn pipeline
        raw = _apply_saved_preprocessing(raw, config=config)

        sfreq = float(raw.info["sfreq"]) # sampling freq
        # crops signal to requested duration
        if seconds is not None and seconds > 0:
            max_samples = int(seconds * sfreq)
            if max_samples > 0 and raw.n_times > max_samples:
                raw.crop(tmin=0.0, tmax=max(0.0, (max_samples - 1) / sfreq))

        data, times = raw.get_data(return_times=True) # extracts sig values and time values
        summary = {
            "recording": recording,
            "n_channels": int(data.shape[0]),
            "n_samples": int(data.shape[1]),
            "duration_sec": float(times[-1]) if len(times) else 0.0,
            "sampling_rate": sfreq,
            "channel_names": list(raw.ch_names),
            "preprocessed": True,
        }
        return data.astype(np.float32), times.astype(np.float32), summary, None
    except Exception as e:
        return None, None, None, f"Failed to load preview for {recording}: {e}"
    finally:
        tmpdir.cleanup()