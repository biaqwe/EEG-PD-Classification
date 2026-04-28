from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent # root directory of the app
RUNS_DIR = APP_DIR / "runs" # stores runs
RESULTS_DIR = APP_DIR / "results" # stores generated outputs

RUNS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLUMNS = ["label", "class", "y", "target"] # possibile names for target col in dataset

META_COLUMNS = [
    "group", # PD or HC
    "subject_id", # subject name
    "subject_key", # subject id
    "window_start", # where the window starts in the signal
    "recording", # recording id
    "part", # segment index
    "start", # start sample
    "source_file", # original file
]

RANDOM_STATE = 42 # sets a fixed random seed for reproducible results
N_SPLITS = 5 # nr of folds for cv
TEST_SIZE = 0.25 # test set size for train/test split
K_BEST = 100 # nr of best feats selected

RAW_WINDOW_SEC = 8.0 # sig window length
RAW_STEP_SEC = 4.0 # step between windows
RAW_MAX_WINDOWS_PER_RECORDING = 60 # max nr of windows from each recording
RAW_L_FREQ = 1.0 # bandpass filter range
RAW_H_FREQ = 35.0 # bandpass filter range
RAW_NOTCH_FREQ = 50.0 # freq removed by notch filter
RAW_USE_BANDPASS = True # bandpass enabled?
RAW_USE_NOTCH = True # notch enabled?
RAW_VAL_SIZE = 0.20 # validation set size for cnn training

CNN_EPOCHS = 60 # nr of training passes thru dataset
CNN_BATCH_SIZE = 16 # nr of samples processed at once
CNN_LR = 3e-4 # learning rate
CNN_WEIGHT_DECAY = 5e-4 # weight decay to prevent overfitting
CNN_DROPOUT = 0.45 # dropout probability to improve generalization
CNN_PATIENCE = 10 # for early stopping

SPECTROGRAM_HEIGHT = 64 # spectrogram image height used by cnn
SPECTROGRAM_WIDTH = 64 # spectrogram image width used by cnn
SPECTROGRAM_NPERSEG = 128 # nr of samples per spectrogram segment
SPECTROGRAM_OVERLAP_RATIO = 0.75 # overlap between spectrogram segments
SPECTROGRAM_CHANNEL_MODE = "mean" # mean uses one averaged spectrogram instead of 63 separate input channels