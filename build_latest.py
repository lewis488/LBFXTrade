import os
import json
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# =========================
# ENV / CONFIG
# =========================

FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY not found in environment variables")

OUTPUT_PATH = Path("public/latest.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================
# FRED SERIES (CORE MACRO)
# =========================
# Keep this lean & robust

FRED_SERIES = {
    "real_yields_10y": "DFII10",           # 10Y TIPS (Gold critical)
    "dxy_proxy": "DTWEXBGS",               # Broad USD index
    "sp500_proxy": "SP500",                # S&P500 index
    "vix_proxy": "VIXCLS",                 # Volatility
    "gold_lbma": "GOLDAMGBD228NLBM",        # Gold (may fail intermittently)
}

# =========================
# DATA FETCH
# =========================

def fred_observations(series_id: str, max_points: int = 120) -> pd.Series:
    """
    Fetch FRED series safely.
    - No date params (FRED handles bounds)
    - Fault tolerant
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[WARN] FRED fetch failed for {series_id}: {e}")
        return pd.Series(dtype=float)

    data = r.json()
    obs = data.get("observations", [])

    if not obs:
        return pd.Series(dtype=float)

    obs = obs[-max_points:]

    cleaned = {}
    for o in obs:
        if o.get("value") not in (".", None):
            try:
                cleaned[o["date"]] = float(o["value"])
            except ValueError:
                continue

    return pd.Series(cleaned)

# =========================
# ANALYSIS LOGIC
# =========================

def rate_of_change(series: pd.Series, periods: int = 5) -> float:
    if len(series) <= periods:
        return 0.0
    return series.iloc[-1] - series.iloc[-periods - 1]

def classify_bias(score: float) -> str:
    if score > 0.5:
        return "Bullish"
    if score < -0.5:
        return "Bearish"
    return "Neutral"

# =========================
# MAIN
# =========================

def main():
    # Fetch macro series
    series_data = {}

    for name, sid in FRED_SERIES.items():
        s = fred_observations(sid)
        if not s.empty:
            series_data[name] = s
        else:
            print(f"[INFO] Skipping {name} (no data)")

    # =========================
    # US500 FUNDAMENTALS
    # =========================

    us500_score = 0.0
    us500_drivers = []

    if "real_yields_10y" in series_data:
        roc = rate_of_change(series_data["real_yields_10y"])
        us500_score -= roc
        us500_drivers.append("Real yields easing" if roc < 0 else "Real yields rising")

    if "vix_proxy" in series_data:
        roc = rate_of_change(series_data["vix_proxy"])
        us500_score -= roc * 0.5
        us500_drivers.append("Volatility falling" if roc < 0 else "Volatility elevated")

    us500_bias = classify_bias(us500_score)

    # =========================
    # GOLD FUNDAMENTALS
    # =========================

    gold_score = 0.0
    gold_drivers = []

    if "real_yields_10y" in series_data:
        roc = rate_of_change(series_data["real_yields_10y"])
        gold_score -= roc
        gold_drivers.append("Real yields falling" if roc < 0 else "Real yields rising")

    if "dxy_proxy" in series_data:
        roc = rate_of_change(series_data["dxy_proxy"])
        gold_score -= roc * 0.7
        gold_drivers.append("USD weakening" if roc < 0 else "USD strengthening")

    gold_bias = classify_bias(gold_score)

    # =========================
    # OUTPUT
    # =========================

    payload = {
        "updated_utc": datetime.utcnow().isoformat(),
        "us500": {
            "bias": us500_bias,
            "confidence": "High" if abs(us500_score) > 1 else "Medium",
            "drivers": us500_drivers[:3],
        },
        "gold": {
            "bias": gold_bias,
            "confidence": "High" if abs(gold_score) > 1 else "Medium",
            "drivers": gold_drivers[:3],
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print("[SUCCESS] latest.json updated")

if __name__ == "__main__":
    main()
