from datetime import datetime

def now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default):
    try:
        return int(x)
    except Exception:
        return default