import io
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

RAW_EXTENSIONS = {".vhdr", ".eeg", ".vmrk", ".set", ".fdt"}
METADATA_EXTENSIONS = {".tsv", ".csv"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS.union(METADATA_EXTENSIONS)


def _clean_uploaded_name(name: str): # keeps only the file name
    return Path(str(name).replace("\\", "/")).name


def _normalize_key(value: str): # normalizes subject keys
    s = str(value).strip().lower()
    s = re.sub(r"\.[a-z0-9]+$", "", s)
    m = re.search(r"(sub[-_ ]?\d+)", s)
    if m:
        return re.sub(r"[-_ ]+", "-", m.group(1))
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _label_from_value(value): # tries to guess class label from value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return None

    if text in {"0", "0.0"}: # checks for negative class indicators
        return 0
    if text in {"1", "1.0"}: # checks for positive class indicators
        return 1

    hc_tokens = [
        "hc", "healthy", "control", "controls", "normal", "cn", "td", "non-pd", "non_pd", "no pd"
    ]
    pd_tokens = [
        "pd", "parkinson", "parkinsons", "parkinson's", "patient", "patients", "disease"
    ]

    if any(tok in text for tok in hc_tokens): # checks for negative class indicators in text
        return 0
    if any(tok in text for tok in pd_tokens): # checks for positive class indicators in text
        return 1

    return None


def infer_label_from_name(name: str): # tries to guess class label from filename
    return _label_from_value(name)


def subject_key_from_name(name: str): # extracts subject id from filename
    stem = Path(str(name).replace("\\", "/")).stem.lower()

    m = re.search(r"(sub[-_ ]?\d+)", stem) # looks for patterns
    if m:
        return _normalize_key(m.group(1)) # normalizes the subject key

    m = re.search(r"(?:subject|participant|subj|sub|s)[-_ ]?(\d+)", stem)
    if m:
        return f"sub-{int(m.group(1)):03d}"

    if "_task" in stem: # removes task info if present
        stem = stem.split("_task", 1)[0]

    return _normalize_key(stem)


def _read_metadata_table(filename: str, data: bytes): # reads metadata table
    ext = Path(filename).suffix.lower()
    buffer = io.BytesIO(data)
    if ext == ".tsv":
        return pd.read_csv(buffer, sep="\t")
    return pd.read_csv(buffer)


def _find_column(columns, candidates): # finds the best matching col for subject keys/labels
    lower_to_original = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    for c in columns:
        cl = str(c).strip().lower()
        if any(candidate in cl for candidate in candidates):
            return c
    return None


def _participants_label_map(payloads: dict): # builds a map from subject keys to labels
    subject_candidates = [
        "participant_id", "participant", "subject_id", "subject", "subject_key", "sub", "id"
    ]
    label_candidates = [
        "label", "class", "target", "y", "group", "diagnosis", "diagnostic", "condition",
        "status", "disease", "phenotype", "research_group", "participant_group"
    ]

    # looks for metadata files and tries to build a map
    for filename, data in payloads.items():
        basename = _clean_uploaded_name(filename)
        lower_name = basename.lower()
        if Path(basename).suffix.lower() not in METADATA_EXTENSIONS:
            continue
        if "participant" not in lower_name and "subject" not in lower_name and "metadata" not in lower_name:
            continue

        try:
            df = _read_metadata_table(basename, data)
        except Exception:
            continue

        if df.empty:
            continue

        subject_col = _find_column(df.columns, subject_candidates) or df.columns[0] # if no subject col found takes the first one
        label_col = _find_column(df.columns, label_candidates) # tries to find label col

        # if no label col found tries to find best candidate based on val parsing
        if label_col is None:
            best_col = None
            best_count = 0
            for c in df.columns:
                if c == subject_col:
                    continue
                parsed = df[c].map(_label_from_value)
                count = int(parsed.notna().sum())
                if count > best_count:
                    best_col = c
                    best_count = count
            if best_count > 0:
                label_col = best_col

        if label_col is None: # if no label col found tries to find best candidate based on name
            continue

        mapping = {}
        # builds map
        for _, row in df.iterrows():
            key = _normalize_key(row.get(subject_col, ""))
            label = _label_from_value(row.get(label_col, None))
            if key and label is not None:
                mapping[key] = int(label)

        if mapping:
            return mapping, {
                "metadata_file": basename,
                "subject_column": str(subject_col),
                "label_column": str(label_col),
                "n_labelled_subjects": int(len(mapping)),
            }

    return {}, None


def _lookup_label(label_map: dict, *values): # looks for label in the map using subject key and filename
    for value in values:
        key = _normalize_key(value)
        if key in label_map:
            return int(label_map[key]), "metadata"

    for value in values:
        label = infer_label_from_name(value)
        if label is not None:
            return int(label), "filename"

    return None, "missing"


def build_brainvision_payload(uploaded_files): # processes uploaded raw eeg files
    if not uploaded_files:
        return None, None, "No files uploaded."

    payloads = {} # for file data
    brainvision_grouped = {} # for files belonging to the same brainvision recording
    eeglab_grouped = {} # for files belonging to the same eeglab recording

    for f in uploaded_files:
        name = _clean_uploaded_name(f.name)
        ext = Path(name).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            continue

        data = f.getvalue()
        payloads[name] = data

        stem = Path(name).stem
        if ext in {".vhdr", ".eeg", ".vmrk"}:
            brainvision_grouped.setdefault(stem, set()).add(ext)
        elif ext in {".set", ".fdt"}:
            eeglab_grouped.setdefault(stem, set()).add(ext)

    if not payloads:
        return None, None, "No supported EEG files found. Upload .set/.fdt, .vhdr/.eeg/.vmrk, and optionally participants.tsv."

    label_map, metadata_info = _participants_label_map(payloads)
    rows = []
    valid_recordings = 0

    for stem, exts in sorted(brainvision_grouped.items()):
        has_vhdr = ".vhdr" in exts
        has_eeg = ".eeg" in exts
        has_vmrk = ".vmrk" in exts
        complete = has_vhdr and has_eeg and has_vmrk
        subject_key = subject_key_from_name(stem)
        label, label_source = _lookup_label(label_map, subject_key, stem)

        if complete:
            valid_recordings += 1

        rows.append({ # recording info
            "recording": stem, # recording name
            "source_format": "BrainVision", # format
            "subject_key": subject_key, # subject if
            "label_guess": label, # guessed class from filename
            "label_source": label_source, # how the label was obtained
            "has_vhdr": has_vhdr, # header present
            "has_eeg": has_eeg, # sig present
            "has_vmrk": has_vmrk, # marker present
            "has_set": False, # eeglab header present
            "has_fdt": False, # eeglab data present
            "complete_triplet": complete, # all brainvision files present
            "complete_eeglab_pair": False, # all eeglab files present
            "complete_recording": complete, # complete eeglab recording
        })

    for stem, exts in sorted(eeglab_grouped.items()): # processes eeglab files
        has_set = ".set" in exts
        has_fdt = ".fdt" in exts
        complete = has_set and has_fdt
        subject_key = subject_key_from_name(stem)
        label, label_source = _lookup_label(label_map, subject_key, stem)

        if complete: # counts valid recordings
            valid_recordings += 1

        rows.append({ # recording info
            "recording": stem,
            "source_format": "EEGLAB",
            "subject_key": subject_key,
            "label_guess": label,
            "label_source": label_source,
            "has_vhdr": False,
            "has_eeg": False,
            "has_vmrk": False,
            "has_set": has_set,
            "has_fdt": has_fdt,
            "complete_triplet": complete,
            "complete_eeglab_pair": complete,
            "complete_recording": complete,
        })

    manifest_df = pd.DataFrame(rows) # rows to dataframe

    if manifest_df.empty:
        return None, None, "No valid raw EEG recordings found."

    if valid_recordings == 0:
        return None, None, "No complete recordings found. For Kaggle EEGLAB data, upload matching .set and .fdt files."

    if metadata_info is not None:
        manifest_df.attrs["metadata_info"] = metadata_info

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
        out_path = root / _clean_uploaded_name(filename)
        out_path.write_bytes(data)

    # fixes links for all files in temp folder
    for file_path in root.iterdir():
        if file_path.suffix.lower() in [".vhdr", ".vmrk"]:
            _rewrite_brainvision_links(file_path)

    recording_paths = []

    for path in sorted(root.glob("*.vhdr")): # looks for brainvision header files to identify recordings
        stem = path.stem
        if (root / f"{stem}.eeg").exists() and (root / f"{stem}.vmrk").exists():
            recording_paths.append({
                "path": path,
                "recording": stem,
                "source_format": "BrainVision",
            })

    for path in sorted(root.glob("*.set")): # looks for eeglab header files to identify recordings
        stem = path.stem
        if (root / f"{stem}.fdt").exists():
            recording_paths.append({
                "path": path,
                "recording": stem,
                "source_format": "EEGLAB",
            })

    return tmpdir, recording_paths


def _read_raw_recording(recording_item): # reads a raw eeg recording using mne
    import mne

    path = recording_item["path"]
    source_format = recording_item["source_format"] # identifies format of recording to use correct mne reader

    if source_format == "BrainVision":
        return mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")

    if source_format == "EEGLAB":
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")

    raise ValueError(f"Unsupported raw EEG format: {source_format}")


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

    tmpdir, recording_paths = materialize_brainvision_payload(payloads)
    label_map, metadata_info = _participants_label_map(payloads)

    X_list = [] # eeg windows
    y_list = [] # window labels
    groups = [] # subject ids
    rows = [] # metadata rows
    channels_count = None # expected nr of channels
    channel_names_ref = None # expected channel names
    sfreq_ref = None # sampling freq ref
    missing_label_recordings = [] # recs with no label
    skipped_channel_recordings = [] # recs with unexpected nr of channels

    try: # makes sure temp folder is deleted at the end
        for item in recording_paths:
            stem = item["recording"]
            subject_key = subject_key_from_name(stem)
            label, label_source = _lookup_label(label_map, subject_key, stem)

            if label is None:
                missing_label_recordings.append(stem)
                continue

            try: # loads eeg recording
                raw = _read_raw_recording(item) # preload means sig data is loaded into memory immediately
            except Exception as e:
                return None, None, None, None, None, f"Failed to read {stem}: {e}"

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
                channel_names_ref = list(raw.ch_names)

            if data.shape[0] != channels_count:
                skipped_channel_recordings.append(stem)
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
                    "source_format": item["source_format"],
                    "subject_key": subject_key,
                    "label": int(label),
                    "label_source": label_source,
                    "window_index": int(i),
                    "window_start": int(start),
                    "start_sample": int(start),
                    "end_sample": int(end),
                })

    finally:
        tmpdir.cleanup() # deletes temp folder

    if not X_list:
        if missing_label_recordings:
            sample = ", ".join(missing_label_recordings[:10])
            return None, None, None, None, None, (
                "Could not generate windows because labels were not detected. "
                "Upload the participants.tsv file together with the .set/.fdt files, "
                "or rename recordings to include HC/PD. Missing labels for: " + sample
            )
        return None, None, None, None, None, "Could not generate windows from the uploaded raw EEG files."

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
        "channel_names": channel_names_ref or [],
        "class_counts": {
            "hc": int((y == 0).sum()),
            "pd": int((y == 1).sum()),
        },
        "metadata_info": metadata_info,
        "missing_label_recordings": missing_label_recordings,
        "skipped_channel_recordings": skipped_channel_recordings,
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


def _bandpower_from_psd(freqs, psd, low, high): # calculates power in a freq band from psd
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return np.zeros(psd.shape[0], dtype=np.float32)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=1).astype(np.float32)


def _window_to_feature_row(window: np.ndarray, sfreq: float, channel_names=None): # extracts feats from one eeg window for tabular training
    from scipy.signal import welch

    n_channels = window.shape[0]
    if not channel_names or len(channel_names) != n_channels:
        channel_names = [f"ch_{i + 1:02d}" for i in range(n_channels)]

    freqs, psd = welch(window, fs=sfreq, nperseg=min(512, window.shape[1]), axis=1) # estimates psd for each channel using welch method
    total_power = _bandpower_from_psd(freqs, psd, 0.5, min(45.0, sfreq / 2.0)) + 1e-12

    bands = { # standard eeg freq bands
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, min(45.0, sfreq / 2.0)),
    }

    row = {}
    for idx, ch in enumerate(channel_names):
        safe_ch = re.sub(r"[^a-zA-Z0-9]+", "_", ch).strip("_").lower() or f"ch_{idx + 1:02d}"
        signal = window[idx]

        row[f"{safe_ch}_mean"] = float(np.mean(signal)) # mean amplitude
        row[f"{safe_ch}_std"] = float(np.std(signal)) # amplitude std
        row[f"{safe_ch}_rms"] = float(np.sqrt(np.mean(signal ** 2))) # root mean square
        row[f"{safe_ch}_ptp"] = float(np.ptp(signal)) # peak to peak amplitude

        for band_name, (low, high) in bands.items():
            if high <= low:
                value = 0.0
                rel_value = 0.0
            else:
                value = float(_bandpower_from_psd(freqs, psd[[idx]], low, high)[0])
                rel_value = float(value / total_power[idx])

            row[f"{safe_ch}_{band_name}_power"] = value # absolute power in band
            row[f"{safe_ch}_{band_name}_rel_power"] = rel_value # relative power in band

    return row


def load_brainvision_feature_table(payloads: dict, config: dict | None = None): # preps tabular feats for training traditional ml models
    cfg = _preprocessing_values(config)

    # loads preprocessed eeg windows and extracts feats
    X_windows, y, groups, meta_df, summary, err = load_brainvision_windows(
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
        return None, None, err

    sfreq = float(summary["sampling_rate"])
    channel_names = summary.get("channel_names", [])
    feature_rows = []

    meta_cols = {
        "recording",
        "source_format",
        "subject_key",
        "label",
        "label_source",
        "window_index",
        "window_start",
        "start_sample",
        "end_sample",
    }

    for i, window in enumerate(X_windows):
        row = meta_df.iloc[i].to_dict()
        row.update(_window_to_feature_row(window, sfreq, channel_names))
        row["label"] = int(y[i])
        feature_rows.append(row)

    df = pd.DataFrame(feature_rows)

    feature_summary = dict(summary)
    feature_summary.update({
        "input_type": "tabular_features",
        "n_rows": int(len(df)),
        "n_features": int(len([c for c in df.columns if c not in meta_cols])),
    })

    return df, feature_summary, None


def load_brainvision_recording_preview(payloads: dict, recording: str, seconds: float | None = None, config: dict | None = None): # loads one raw eeg recoring and prepares a preview for visualization
    try:
        import mne
    except Exception as e:
        return None, None, None, f"MNE is not available: {e}"

    tmpdir, recording_paths = materialize_brainvision_payload(payloads)

    try:
        # searches for the selected recording
        selected_item = None
        for item in recording_paths:
            if item["recording"] == recording:
                selected_item = item
                break

        if selected_item is None:
            return None, None, None, f"Recording not found: {recording}"

        raw = _read_raw_recording(selected_item) # reads the file using mne
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
            "source_format": selected_item["source_format"],
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