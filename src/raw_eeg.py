import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def infer_label_from_name(name: str):
    s = name.lower()

    if "control" in s or "_hc" in s or "-hc" in s or s.startswith("hc"):
        return 0

    if "pd" in s or "parkinson" in s or "patient" in s:
        return 1

    return None


def subject_key_from_name(name: str):
    stem = Path(name).stem
    m = re.search(r"([A-Za-z]+)(\d+)", stem)
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"
    return stem.lower()


def build_brainvision_payload(uploaded_files):
    if not uploaded_files:
        return None, None, "No files uploaded."

    payloads = {}
    grouped = {}

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
        complete = has_vhdr and has_eeg and has_vmrk

        label = infer_label_from_name(stem)
        subject_key = subject_key_from_name(stem)

        if complete:
            valid_recordings += 1

        rows.append({
            "recording": stem,
            "subject_key": subject_key,
            "label_guess": label,
            "has_vhdr": has_vhdr,
            "has_eeg": has_eeg,
            "has_vmrk": has_vmrk,
            "complete_triplet": complete,
        })

    manifest_df = pd.DataFrame(rows)

    if manifest_df.empty:
        return None, None, "No valid BrainVision files found."

    if valid_recordings == 0:
        return None, None, "No complete BrainVision triplets found (.vhdr + .eeg + .vmrk)."

    return payloads, manifest_df, None


def _rewrite_brainvision_links(file_path: Path):
    ext = file_path.suffix.lower()
    stem = file_path.stem

    if ext not in [".vhdr", ".vmrk"]:
        return

    text = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    new_lines = []
    for line in lines:
        if line.startswith("DataFile="):
            new_lines.append(f"DataFile={stem}.eeg")
        elif line.startswith("MarkerFile="):
            new_lines.append(f"MarkerFile={stem}.vmrk")
        else:
            new_lines.append(line)

    file_path.write_text("\n".join(new_lines), encoding="utf-8")


def materialize_brainvision_payload(payloads: dict):
    tmpdir = tempfile.TemporaryDirectory()
    root = Path(tmpdir.name)

    for filename, data in payloads.items():
        out_path = root / Path(filename).name
        out_path.write_bytes(data)

    for file_path in root.iterdir():
        if file_path.suffix.lower() in [".vhdr", ".vmrk"]:
            _rewrite_brainvision_links(file_path)

    vhdr_paths = sorted(root.glob("*.vhdr"))
    return tmpdir, vhdr_paths


def load_brainvision_windows(
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

    X_list = []
    y_list = []
    groups = []
    rows = []
    channels_count = None
    sfreq_ref = None

    try:
        for vhdr_path in vhdr_paths:
            stem = vhdr_path.stem
            label = infer_label_from_name(stem)

            if label is None:
                continue

            subject_key = subject_key_from_name(stem)

            try:
                raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
            except Exception as e:
                return None, None, None, None, None, f"Failed to read {vhdr_path.name}: {e}"

            raw.pick("eeg")

            if len(raw.ch_names) == 0:
                continue

            if use_bandpass:
                raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")

            if use_notch and notch_freq > 0:
                raw.notch_filter(freqs=[notch_freq], verbose="ERROR")

            data = raw.get_data().astype(np.float32)
            sfreq = float(raw.info["sfreq"])

            if sfreq_ref is None:
                sfreq_ref = sfreq

            if channels_count is None:
                channels_count = data.shape[0]

            if data.shape[0] != channels_count:
                continue

            win_samples = int(window_sec * sfreq)
            step_samples = int(step_sec * sfreq)

            if win_samples <= 0 or step_samples <= 0:
                continue

            total = data.shape[1]
            starts = list(range(0, max(0, total - win_samples + 1), step_samples))

            if max_windows_per_recording is not None and len(starts) > max_windows_per_recording:
                starts = starts[:max_windows_per_recording]

            for i, start in enumerate(starts):
                end = start + win_samples
                window = data[:, start:end]

                if window.shape[1] != win_samples:
                    continue

                X_list.append(window)
                y_list.append(label)
                groups.append(subject_key)

                rows.append({
                    "recording": stem,
                    "subject_key": subject_key,
                    "label": int(label),
                    "window_index": int(i),
                    "start_sample": int(start),
                    "end_sample": int(end),
                })

    finally:
        tmpdir.cleanup()

    if not X_list:
        return None, None, None, None, None, "Could not generate windows from the uploaded BrainVision files."

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(groups)
    meta_df = pd.DataFrame(rows)

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