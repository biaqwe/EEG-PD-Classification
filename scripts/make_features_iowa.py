import h5py
import numpy as np
import pandas as pd

PATH = "/Users/bianca/Desktop/licenta/Data and Code (1)/Dataset/IowaDataset/Organized data/IowaData.mat"

WINDOW = 2000
STEP = 2000
MAX_WINDOWS_PER_SUBJECT = 30

GROUP0_LABEL = 1  # PD
GROUP1_LABEL = 0  # Control

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
    parts = np.array(ch_obj[()]).reshape(-1)  # 2 refs (PD/Control groups)
    group_obj = f[parts[group_idx]]
    subj_refs = np.array(group_obj[()]).reshape(-1)  # 14 refs (subjects)
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

def features_1d(x: np.ndarray):
    x = x.astype(np.float32)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ptp": float(np.ptp(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "zcr": float(np.mean(np.abs(np.diff(np.sign(x))) > 0)),
    }

def get_subject_matrix(f, eeg_refs, valid_channels, group_idx, subj_idx):
    signals = []
    min_len = None

    for ch in valid_channels:
        ch_ref = eeg_refs[ch]
        subj_refs = get_subject_refs_for_channel(f, ch_ref, group_idx)
        sig = np.array(f[subj_refs[subj_idx]][()]).squeeze().astype(np.float32)

        signals.append(sig)
        min_len = sig.size if min_len is None else min(min_len, sig.size)

    X = np.stack([s[:min_len] for s in signals], axis=0)  # (n_valid_channels, T)
    return X

with h5py.File(PATH, "r") as f:
    eeg_refs = np.array(f["EEG"][()]).reshape(-1)

    fn_refs = np.array(f["Filenames"][()]).reshape(-1)
    pd_ids = decode_strings_list(f, f[fn_refs[0]])
    hc_ids = decode_strings_list(f, f[fn_refs[1]])

    # n_groups / n_subj
    ch0 = f[eeg_refs[0]]
    parts0 = np.array(ch0[()]).reshape(-1)
    n_groups = len(parts0)

    group0_obj = f[parts0[0]]
    n_subj = len(np.array(group0_obj[()]).reshape(-1))

    print("n_groups:", n_groups)
    print("n_subj:", n_subj)
    print("PD ids:", len(pd_ids), "HC ids:", len(hc_ids))

    valid_channels, invalid_examples = find_valid_channels(
        f=f,
        eeg_refs=eeg_refs,
        n_groups=n_groups,
        n_subj=n_subj,
        min_len=WINDOW,
    )

    print("Total channels:", len(eeg_refs))
    print("Valid channels (len >= WINDOW for all subjects):", len(valid_channels))
    if len(valid_channels) == 0:
        print("No valid channels found. Example invalid entries (ch, group, subj, dtype, shape):")
        for ex in invalid_examples[:10]:
            print(" ", ex)
        raise SystemExit(1)

    rows = []

    for group_idx in range(n_groups):
        if group_idx == 0:
            label = GROUP0_LABEL
            subject_ids = pd_ids
        else:
            label = GROUP1_LABEL
            subject_ids = hc_ids

        for subj_idx in range(n_subj):
            subj_id = subject_ids[subj_idx]

            X = get_subject_matrix(f, eeg_refs, valid_channels, group_idx, subj_idx)
            T = X.shape[1]

            n_w = 0
            for start in range(0, T - WINDOW + 1, STEP):
                if n_w >= MAX_WINDOWS_PER_SUBJECT:
                    break

                seg = X[:, start:start + WINDOW]  # (n_valid_channels, WINDOW)

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
                n_w += 1

df = pd.DataFrame(rows)
df.to_csv("dataset_iowa_pd_hc.csv", index=False)

print("Saved: dataset_iowa_pd_hc.csv")
print("Rows:", len(df), "Cols:", len(df.columns))
print("Label counts:\n", df["label"].value_counts())
print("Unique subject_id:", df["subject_id"].nunique())
print("Unique subject_key:", df["subject_key"].nunique())
print("First subject_id values:", df["subject_id"].unique()[:10])
