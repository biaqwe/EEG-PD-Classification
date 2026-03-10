from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = APP_DIR / "runs"
RESULTS_DIR = APP_DIR / "results"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLUMNS = ["label", "class", "y", "target"]

META_COLUMNS = [
    "group",
    "subject_id",
    "subject_key",
    "window_start",
    "recording",
    "part",
    "start",
    "source_file",
]

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.25
K_BEST = 100