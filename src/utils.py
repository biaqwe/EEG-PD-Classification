from datetime import datetime

def now_iso(): # returns current timestamp as formatted string
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_float(x, default): # safely converts value to float
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default): # safely converts value to int
    try:
        return int(x)
    except Exception:
        return default