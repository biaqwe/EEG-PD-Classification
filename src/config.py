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

RAW_WINDOW_SEC = 2.0
RAW_STEP_SEC = 2.0
RAW_MAX_WINDOWS_PER_RECORDING = 30
RAW_L_FREQ = 0.5
RAW_H_FREQ = 40.0
RAW_NOTCH_FREQ = 50.0
RAW_USE_BANDPASS = True
RAW_USE_NOTCH = True
RAW_VAL_SIZE = 0.20

CNN_EPOCHS = 20
CNN_BATCH_SIZE = 32
CNN_LR = 1e-3
CNN_WEIGHT_DECAY = 1e-4
CNN_DROPOUT = 0.25
CNN_PATIENCE = 5