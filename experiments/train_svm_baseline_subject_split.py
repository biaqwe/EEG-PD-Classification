import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

CSV_PATH = "dataset_iowa_pd_hc.csv"
TEST_SIZE = 0.30
RANDOM_STATE = 42

def main():
    df = pd.read_csv(CSV_PATH)

    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column.")
    if "subject_key" not in df.columns:
        raise ValueError("CSV must contain 'subject_key' column for subject-level split.")

    meta_cols = {"label", "group", "subject_id", "subject_key", "window_start"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].astype(np.float32).values
    y = df["label"].astype(int).values
    groups = df["subject_key"].astype(str).values

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups.intersection(test_groups)

    print("Rows:", df.shape[0], "Features:", len(feature_cols))
    print("Subjects total:", len(set(groups)))
    print("Train subjects:", len(train_groups), "Test subjects:", len(test_groups))
    print("Subject overlap (must be 0):", len(overlap))

    print("\nTrain label counts:")
    print(pd.Series(y_train).value_counts().sort_index())
    print("\nTest label counts:")
    print(pd.Series(y_test).value_counts().sort_index())

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)

    auc = None
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        pass

    print("\n=== Metrics (subject-level split) ===")
    print("Accuracy:", round(float(acc), 4))
    print("F1:", round(float(f1), 4))
    if auc is not None:
        print("AUC:", round(float(auc), 4))

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    out = df.iloc[test_idx][["subject_id", "subject_key", "label"]].copy()
    out["proba_pd"] = proba
    out["pred"] = pred
    out.to_csv("svm_test_predictions.csv", index=False)
    print("\nSaved: svm_test_predictions.csv")

if __name__ == "__main__":
    main()
