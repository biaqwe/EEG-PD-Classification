from pathlib import Path
import argparse

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

DEFAULT_FS = 1000.0
DEFAULT_WINDOW = 2000
DEFAULT_STEP = 2000
DEFAULT_MAX_WINDOWS_PER_SUBJECT = 30

DEFAULT_BANDPASS_LOW = 0.5
DEFAULT_BANDPASS_HIGH = 40.0
DEFAULT_NOTCH_FREQ = 50.0
DEFAULT_NOTCH_Q = 30.0
DEFAULT_FILTER_ORDER = 4

GROUP0_LABEL = 1   # PD
GROUP1_LABEL = 0   # HC / Control


def decode_uint16(ds):
    arr = np.array(ds[()]).squeeze()
    return "".join(chr(int(c)) for c in arr if int(c) != 0).strip()


def decode_strings_list(f, obj):
    refs = np.array(obj[()]).reshape(-1)
    return [decode_uint16(f[r]) for r in refs]


def is_numeric_signal(ds, min_len):
    if not isinstance(ds, h5py.Dataset):
        return False
    if ds.dtype.kind not in ("f", "i"):
        return False

    shape = ds.shape
    if shape is None or len(shape) == 0:
        return False

    size = int(np.prod(shape))
    return size >= min_len


def get_subject_refs_for_channel(f, eeg_ref, group_idx):
    ch_obj = f[eeg_ref]
    parts = np.array(ch_obj[()]).reshape(-1)
    group_obj = f[parts[group_idx]]
    subj_refs = np.array(group_obj[()]).reshape(-1)
    return subj_refs


def find_valid_channels(f, eeg_refs, n_groups, n_subj, min_len):
    valid = []
    invalid_info = []

    for ch_idx, ch_ref in enumerate(eeg_refs):
        ok = True
        bad_example = None

        for g in range(n_groups):
            subj_refs = get_subject_refs_for_channel(f, ch_ref, g)

            for s in range(n_subj):
                ds = f[subj_refs[s]]
                if not is_numeric_signal(ds, min_len):
                    ok = False
                    bad_example = (ch_idx, g, s, str(ds.dtype), str(ds.shape))
                    break

            if not ok:
                break

        if ok:
            valid.append(ch_idx)
        else:
            invalid_info.append(bad_example)

    return valid, invalid_info


def bandpass_filter(
    x,
    fs,
    low=DEFAULT_BANDPASS_LOW,
    high=DEFAULT_BANDPASS_HIGH,
    order=DEFAULT_FILTER_ORDER,
):
    x = np.asarray(x, dtype=np.float32)

    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq

    if not (0 < low_norm < high_norm < 1):
        raise ValueError(f"Invalid bandpass range: low={low}, high={high}, fs={fs}")

    b, a = butter(order, [low_norm, high_norm], btype="band")
    y = filtfilt(b, a, x)
    return y.astype(np.float32)


def notch_filter(x, fs, freq=DEFAULT_NOTCH_FREQ, q=DEFAULT_NOTCH_Q):
    x = np.asarray(x, dtype=np.float32)

    nyq = 0.5 * fs
    w0 = freq / nyq

    if not (0 < w0 < 1):
        raise ValueError(f"Invalid notch frequency: freq={freq}, fs={fs}")

    b, a = iirnotch(w0, q)
    y = filtfilt(b, a, x)
    return y.astype(np.float32)


def preprocess_signal(
    x,
    fs=DEFAULT_FS,
    use_bandpass=False,
    use_notch=True,
    bandpass_low=DEFAULT_BANDPASS_LOW,
    bandpass_high=DEFAULT_BANDPASS_HIGH,
    notch_freq=DEFAULT_NOTCH_FREQ,
    notch_q=DEFAULT_NOTCH_Q,
    filter_order=DEFAULT_FILTER_ORDER,
):
    y = np.asarray(x, dtype=np.float32)

    if use_bandpass:
        y = bandpass_filter(
            y,
            fs=fs,
            low=bandpass_low,
            high=bandpass_high,
            order=filter_order,
        )

    if use_notch:
        y = notch_filter(
            y,
            fs=fs,
            freq=notch_freq,
            q=notch_q,
        )

    return y.astype(np.float32)


def zero_crossing_rate(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size < 2:
        return 0.0
    return float(np.mean(np.diff(np.signbit(x)) != 0))


def features_1d(x):
    x = np.asarray(x, dtype=np.float32)

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ptp": float(np.ptp(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "zcr": float(zero_crossing_rate(x)),
    }


def get_subject_matrix(
    f,
    eeg_refs,
    valid_channels,
    group_idx,
    subj_idx,
    fs=DEFAULT_FS,
    use_bandpass=False,
    use_notch=True,
    bandpass_low=DEFAULT_BANDPASS_LOW,
    bandpass_high=DEFAULT_BANDPASS_HIGH,
    notch_freq=DEFAULT_NOTCH_FREQ,
    notch_q=DEFAULT_NOTCH_Q,
    filter_order=DEFAULT_FILTER_ORDER,
):
    signals = []
    min_len = None

    for ch in valid_channels:
        ch_ref = eeg_refs[ch]
        subj_refs = get_subject_refs_for_channel(f, ch_ref, group_idx)

        sig = np.array(f[subj_refs[subj_idx]][()]).squeeze().astype(np.float32)

        sig = preprocess_signal(
            sig,
            fs=fs,
            use_bandpass=use_bandpass,
            use_notch=use_notch,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            notch_freq=notch_freq,
            notch_q=notch_q,
            filter_order=filter_order,
        )

        signals.append(sig)
        min_len = sig.size if min_len is None else min(min_len, sig.size)

    X = np.stack([s[:min_len] for s in signals], axis=0)
    return X


def build_iowa_features_from_mat(
    mat_path,
    fs=DEFAULT_FS,
    window=DEFAULT_WINDOW,
    step=DEFAULT_STEP,
    max_windows_per_subject=DEFAULT_MAX_WINDOWS_PER_SUBJECT,
    use_bandpass=False,
    use_notch=True,
    bandpass_low=DEFAULT_BANDPASS_LOW,
    bandpass_high=DEFAULT_BANDPASS_HIGH,
    notch_freq=DEFAULT_NOTCH_FREQ,
    notch_q=DEFAULT_NOTCH_Q,
    filter_order=DEFAULT_FILTER_ORDER,
    verbose=True,
):
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Input file not found: {mat_path}")

    rows = []

    with h5py.File(mat_path, "r") as f:
        eeg_refs = np.array(f["EEG"][()]).reshape(-1)

        fn_refs = np.array(f["Filenames"][()]).reshape(-1)
        pd_ids = decode_strings_list(f, f[fn_refs[0]])
        hc_ids = decode_strings_list(f, f[fn_refs[1]])

        ch0 = f[eeg_refs[0]]
        parts0 = np.array(ch0[()]).reshape(-1)
        n_groups = len(parts0)

        group0_obj = f[parts0[0]]
        n_subj = len(np.array(group0_obj[()]).reshape(-1))

        if verbose:
            print("=========================================================")
            print("Iowa feature extraction started")
            print("=========================================================")
            print(f"Input file: {mat_path}")
            print(f"Sampling rate: {fs} Hz")
            print(f"Window: {window} samples ({window / fs:.2f} sec)")
            print(f"Step: {step} samples ({step / fs:.2f} sec)")
            print(f"Max windows / subject: {max_windows_per_subject}")
            print(f"Bandpass enabled: {use_bandpass}")
            print(f"Notch enabled: {use_notch}")
            if use_bandpass:
                print(f"Bandpass: {bandpass_low} - {bandpass_high} Hz")
            if use_notch:
                print(f"Notch: {notch_freq} Hz")
            print("---------------------------------------------------------")
            print("n_groups:", n_groups)
            print("n_subj:", n_subj)
            print("PD ids:", len(pd_ids), "HC ids:", len(hc_ids))
            print("Total raw channels:", len(eeg_refs))

        valid_channels, invalid_examples = find_valid_channels(
            f=f,
            eeg_refs=eeg_refs,
            n_groups=n_groups,
            n_subj=n_subj,
            min_len=window,
        )

        if verbose:
            print("Valid channels:", len(valid_channels))

        if len(valid_channels) == 0:
            if verbose:
                print("No valid channels found.")
                print("Example invalid entries (ch, group, subj, dtype, shape):")
                for ex in invalid_examples[:10]:
                    print(" ", ex)
            raise ValueError("No valid channels found for the selected window length.")

        if verbose:
            print("First valid channels:", valid_channels[:10])
            print("---------------------------------------------------------")

        for group_idx in range(n_groups):
            if group_idx == 0:
                label = GROUP0_LABEL
                subject_ids = pd_ids
                group_name = "PD"
            else:
                label = GROUP1_LABEL
                subject_ids = hc_ids
                group_name = "HC"

            if verbose:
                print(f"Processing group {group_idx} ({group_name})...")

            for subj_idx in range(n_subj):
                subj_id = subject_ids[subj_idx]

                X = get_subject_matrix(
                    f=f,
                    eeg_refs=eeg_refs,
                    valid_channels=valid_channels,
                    group_idx=group_idx,
                    subj_idx=subj_idx,
                    fs=fs,
                    use_bandpass=use_bandpass,
                    use_notch=use_notch,
                    bandpass_low=bandpass_low,
                    bandpass_high=bandpass_high,
                    notch_freq=notch_freq,
                    notch_q=notch_q,
                    filter_order=filter_order,
                )

                T = X.shape[1]
                n_possible = max(0, (T - window) // step + 1)
                n_used = 0

                for start in range(0, T - window + 1, step):
                    if n_used >= max_windows_per_subject:
                        break

                    seg = X[:, start:start + window]

                    feat = {}
                    for i, ch in enumerate(valid_channels):
                        fch = features_1d(seg[i])
                        for k, v in fch.items():
                            feat[f"ch{ch}_{k}"] = v

                    feat["label"] = int(label)
                    feat["group"] = int(group_idx)
                    feat["subject_id"] = str(subj_id)
                    feat["subject_key"] = f"{group_idx}_{subj_idx}"
                    feat["window_start"] = int(start)

                    rows.append(feat)
                    n_used += 1

                if verbose:
                    print(
                        f"  subject {subj_idx:02d} | id={subj_id} | "
                        f"signal_shape={X.shape} | windows_possible={n_possible} | windows_used={n_used}"
                    )

    df = pd.DataFrame(rows)

    summary = {
        "input_file": str(mat_path),
        "fs": fs,
        "window": window,
        "step": step,
        "window_sec": float(window / fs),
        "step_sec": float(step / fs),
        "max_windows_per_subject": max_windows_per_subject,
        "use_bandpass": bool(use_bandpass),
        "use_notch": bool(use_notch),
        "bandpass_low": bandpass_low,
        "bandpass_high": bandpass_high,
        "notch_freq": notch_freq,
        "valid_channels": len(valid_channels),
        "n_groups": n_groups,
        "n_subjects_per_group": n_subj,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
    }

    return df, summary


def save_iowa_features_csv(df, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, default=None, help="Path to IowaData.mat")
    parser.add_argument("--output", type=str, required=False, default="dataset_iowa_pd_hc.csv")
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--max_windows", type=int, default=DEFAULT_MAX_WINDOWS_PER_SUBJECT)
    parser.add_argument("--use_bandpass", action="store_true")
    parser.add_argument("--no_notch", action="store_true")
    parser.add_argument("--bandpass_low", type=float, default=DEFAULT_BANDPASS_LOW)
    parser.add_argument("--bandpass_high", type=float, default=DEFAULT_BANDPASS_HIGH)
    parser.add_argument("--notch_freq", type=float, default=DEFAULT_NOTCH_FREQ)
    parser.add_argument("--notch_q", type=float, default=DEFAULT_NOTCH_Q)

    args = parser.parse_args()

    if args.input is None:
        raise ValueError("Please provide --input path_to_IowaData.mat")

    df, summary = build_iowa_features_from_mat(
        mat_path=args.input,
        fs=args.fs,
        window=args.window,
        step=args.step,
        max_windows_per_subject=args.max_windows,
        use_bandpass=args.use_bandpass,
        use_notch=not args.no_notch,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        notch_freq=args.notch_freq,
        notch_q=args.notch_q,
        verbose=True,
    )

    out_path = save_iowa_features_csv(df, args.output)

    print("=========================================================")
    print(f"Saved: {out_path}")
    print(f"Rows: {len(df)} | Cols: {len(df.columns)}")
    print("Label counts:")
    print(df["label"].value_counts(dropna=False))
    print("Unique subject_id:", df["subject_id"].nunique())
    print("Unique subject_key:", df["subject_key"].nunique())
    print("Summary:")
    print(summary)
    print("=========================================================")


if __name__ == "__main__":
    main()