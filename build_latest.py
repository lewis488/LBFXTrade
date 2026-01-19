import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# =========================
# ENV
# =========================

FRED_API_KEY = os.getenv("FRED_API_KEY")
TE_API_KEY = os.getenv("TE_API_KEY")

if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY missing")

OUTPUT_PATH = Path("latest.json")

# =========================
# FRED SERIES
# =========================

FRED_SERIES = {
    "real_yields": "DFII10",
    "usd": "DTWEXBGS",
    "vix": "VIXCLS",
    "sp500": "SP500",
    "gold": "GOLDAMGBD228NLBM",
}

# =========================
# DATA FETCH
# =========================

def fred_series(series_id, points=120):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return pd.Series(dtype=float)

    obs = r.json().get("observations", [])
    obs = obs[-points:]

    data = {}
    for o in obs:
        if o["value"] not in (".", None):
            try:
                data[o["date"]] = float(o["value"])
            except ValueError:
                pass

    return pd.Series(data)

def roc(series, n=5):
    if len(series) <= n:
        return 0.0
    return series.iloc[-1] - series.iloc[-n - 1]

# =========================
# MACRO LOGIC
# =========================

def classify(score):
    if score > 0.5:
        return "Bullish"
    if score < -0.5:
        return "Bearish"
    return "Neutral"

def risk_regime(series):
    score = 0
    score -= roc(series.get("real_yields", pd.Series()))
    score -= roc(series.get("vix", pd.Series())) * 0.5
    score -= roc(series.get("usd", pd.Series())) * 0.7

    if score > 0.5:
        return "Risk-On"
    if score < -0.5:
        return "Risk-Off"
    return "Mixed"

# =========================
# CPI / FOMC / NFP
# =========================

def detect_macro_risk():
    if not TE_API_KEY:
        return {
            "risk": False,
            "message": "Macro calendar not connected"
        }

    try:
        url = "https://api.tradingeconomics.com/calendar"
        params = {
            "c": TE_API_KEY,
            "country": "United States"
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return {
            "risk": False,
            "message": "Macro calendar unavailable"
        }

    now = datetime.utcnow()
    window = now + timedelta(hours=24)

    HIGH_IMPACT = ["CPI", "Fed", "FOMC", "Nonfarm Payrolls"]

    for ev in r.json():
        name = ev.get("Event", "")
        date_str = ev.get("Date")

        if not date_str:
            continue

        try:
            ev_date = datetime.fromisoformat(date_str.replace("Z",""))
        except ValueError:
            continue

        if now <= ev_date <= window:
            if any(k.lower() in name.lower() for k in HIGH_IMPACT):
                return {
                    "risk": True,
                    "event": name,
                    "message": f"High-impact risk: {name} within 24h"
                }

    return {
        "risk": False,
        "message": "No high-impact macro risk in next 24h"
    }

# =========================
# MAIN
# =========================

def main():
    series = {k: fred_series(v) for k, v in FRED_SERIES.items()}

    # -------- US500 --------
    us_score = 0
    us_drivers = []

    ry = roc(series["real_yields"])
    vix = roc(series["vix"])

    us_score -= ry
    us_score -= vix * 0.5

    us_drivers.append("Real yields easing" if ry < 0 else "Real yields rising")
    us_drivers.append("Volatility elevated" if vix > 0 else "Volatility falling")

    us_bias = classify(us_score)

    # -------- GOLD --------
    g_score = 0
    g_drivers = []

    usd = roc(series["usd"])
    g_score -= ry
    g_score -= usd * 0.7

    g_drivers.append("Real yields falling" if ry < 0 else "Real yields rising")
    g_drivers.append("USD weakening" if usd < 0 else "USD strengthening")

    g_bias = classify(g_score)

    payload = {
        "updated_utc": datetime.utcnow().isoformat(),
        "macro_risk": detect_macro_risk(),
        "risk_regime": risk_regime(series),
        "us500": {
            "bias": us_bias,
            "confidence": "High" if abs(us_score) > 1 else "Medium",
            "drivers": us_drivers[:3],
        },
        "gold": {
            "bias": g_bias,
            "confidence": "High" if abs(g_score) > 1 else "Medium",
            "drivers": g_drivers[:3],
        }
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
