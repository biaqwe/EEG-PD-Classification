import json
from dataclasses import asdict
from datetime import datetime

import streamlit as st

from src.config import RUNS_DIR
from src.data_utils import dataset_summary
from src.state import RunRecord
from src.utils import now_iso

def save_run(action: str, status: str, metrics: dict): # saves info about a training run
    df = st.session_state.dataset_df # gets currently loaded dataset
    n_rows, n_channels = dataset_summary(df) # computes nr of rows and channels
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # generates run id

    rec = RunRecord( # contains all run info
        run_id=run_id,
        timestamp=now_iso(),
        dataset_name=st.session_state.dataset_name,
        n_rows=n_rows,
        n_channels=n_channels,
        preproc=asdict(st.session_state.preproc),
        action=action,
        status=status,
        metrics=metrics or {},
    )

    path = RUNS_DIR / f"{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(rec), f, ensure_ascii=False, indent=2)

    return path

def load_runs(limit: int = 30): # loads previous runs
    items = sorted(RUNS_DIR.glob("*.json"), reverse=True)
    out = []

    for p in items[:limit]:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass

    return out