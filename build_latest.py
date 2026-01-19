import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
TE_API_KEY = os.getenv("TE_API_KEY", "").strip()
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

OUT_PATH = os.path.join("public", "latest.json")

FRED_SERIES = {
    "sp500": "SP500",
    "gold": "GOLDAMGBD228NLBM",
    "dgs10": "DGS10",
    "dfii10": "DFII10",
    "vix": "VIXCLS",
    "usd_broad": "DTWEXBGS",
}


def fred_observations(series_id: str, days_back: int = 120) -> pd.Series:
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is required. Get a free key from FRED.")

    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=days_back)

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    obs = data.get("observations", [])
    if not obs:
        raise RuntimeError(f"No observations returned for {series_id}")

    rows = []
    for o in obs:
        v = o.get("value")
        if v in (".", None):
            continue
        try:
            rows.append((o["date"], float(v)))
        except ValueError:
            continue

    s = pd.Series({pd.to_datetime(d): v for d, v in rows}).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def pct_change(s: pd.Series, periods: int) -> Optional[float]:
    if len(s) <= periods:
        return None
    a = float(s.iloc[-1])
    b = float(s.iloc[-1 - periods])
    if b == 0:
        return None
    return (a - b) / b * 100.0


def abs_change(s: pd.Series, periods: int) -> Optional[float]:
    if len(s) <= periods:
        return None
    return float(s.iloc[-1] - s.iloc[-1 - periods])


def trend_tag(change: Optional[float], pos: str = "up", neg: str = "down") -> str:
    if change is None:
        return "n/a"
    if change > 0:
        return pos
    if change < 0:
        return neg
    return "flat"


def compute_signals(series: Dict[str, pd.Series]) -> Dict[str, Any]:
    # 5-day and 20-day = about 1 week / 1 month trading days
    return {
        "levels": {k: float(v.iloc[-1]) for k, v in series.items()},
        "chg_5d": {
            "sp500": pct_change(series["sp500"], 5),
            "gold": pct_change(series["gold"], 5),
            "usd_broad": pct_change(series["usd_broad"], 5),
            "vix": pct_change(series["vix"], 5),
            "dgs10": abs_change(series["dgs10"], 5),
            "dfii10": abs_change(series["dfii10"], 5),
        },
        "chg_20d": {
            "sp500": pct_change(series["sp500"], 20),
            "gold": pct_change(series["gold"], 20),
            "usd_broad": pct_change(series["usd_broad"], 20),
            "vix": pct_change(series["vix"], 20),
            "dgs10": abs_change(series["dgs10"], 20),
            "dfii10": abs_change(series["dfii10"], 20),
        },
    }


def score_bias(signals: Dict[str, Any]) -> Dict[str, Any]:
    # Simple, explainable scoring. Outputs are *context*, not signals.
    ch5 = signals["chg_5d"]
    ch20 = signals["chg_20d"]

    # US500 drivers
    sp_trend = 1 if (ch20["sp500"] or 0) > 0 else (-1 if (ch20["sp500"] or 0) < 0 else 0)
    vol = 1 if (ch20["vix"] or 0) < 0 else (-1 if (ch20["vix"] or 0) > 0 else 0)
    yields = 1 if (ch20["dgs10"] or 0) < 0 else (-1 if (ch20["dgs10"] or 0) > 0 else 0)

    us500_score = sp_trend + vol + yields

    # Gold drivers: real yields + USD + risk tone
    real_yields = 1 if (ch20["dfii10"] or 0) < 0 else (-1 if (ch20["dfii10"] or 0) > 0 else 0)
    usd = 1 if (ch20["usd_broad"] or 0) < 0 else (-1 if (ch20["usd_broad"] or 0) > 0 else 0)
    risk_off = 1 if (ch20["vix"] or 0) > 0 else (-1 if (ch20["vix"] or 0) < 0 else 0)

    gold_score = real_yields + usd + risk_off

    def label(score: int) -> Tuple[str, str]:
        if score >= 2:
            return "bullish", "high"
        if score == 1:
            return "bullish", "medium"
        if score == 0:
            return "neutral", "medium"
        if score == -1:
            return "bearish", "medium"
        return "bearish", "high"

    us500_bias, us500_conf = label(us500_score)
    gold_bias, gold_conf = label(gold_score)

    # Delta notes
    deltas = {
        "usd": trend_tag(ch5["usd_broad"], pos="strengthening", neg="weakening"),
        "real_yields": trend_tag(ch5["dfii10"], pos="rising", neg="falling"),
        "vix": trend_tag(ch5["vix"], pos="rising", neg="falling"),
    }

    return {
        "us500": {"bias": us500_bias, "confidence": us500_conf, "score": us500_score},
        "xauusd": {"bias": gold_bias, "confidence": gold_conf, "score": gold_score},
        "deltas": deltas,
    }


def fetch_calendar(start: datetime, end: datetime) -> List[Dict[str, Any]]:
    # Returns a small list of high-impact US events.
    # If no calendar API keys, returns empty.
    events: List[Dict[str, Any]] = []

    if TE_API_KEY:
        # Trading Economics: https://docs.tradingeconomics.com/economic_calendar/snapshot/
        url = "https://api.tradingeconomics.com/calendar/country/united%20states"
        params = {"c": TE_API_KEY, "d1": start.date().isoformat(), "d2": end.date().isoformat()}
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            for e in data:
                events.append({
                    "datetime": e.get("Date") or e.get("date"),
                    "event": e.get("Event") or e.get("event"),
                    "importance": e.get("Importance") or e.get("importance"),
                    "actual": e.get("Actual") or e.get("actual"),
                    "forecast": e.get("Forecast") or e.get("forecast"),
                    "previous": e.get("Previous") or e.get("previous"),
                    "country": e.get("Country") or e.get("country"),
                })
        except Exception:
            pass

    elif FMP_API_KEY:
        # FMP endpoint shown in docs: https://financialmodelingprep.com/stable/economic-calendar
        url = "https://financialmodelingprep.com/stable/economic-calendar"
        params = {"from": start.date().isoformat(), "to": end.date().isoformat(), "apikey": FMP_API_KEY}
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            for e in data:
                events.append({
                    "datetime": e.get("date"),
                    "event": e.get("event"),
                    "importance": e.get("impact") or e.get("importance"),
                    "actual": e.get("actual"),
                    "forecast": e.get("estimate") or e.get("forecast"),
                    "previous": e.get("previous"),
                    "country": e.get("country"),
                })
        except Exception:
            pass

    # Keep only future-ish events and trim
    def parse_dt(x: Any) -> Optional[datetime]:
        if not x:
            return None
        try:
            # Trading Economics often uses ISO-ish strings
            dt = pd.to_datetime(x, utc=True)
            if pd.isna(dt):
                return None
            return dt.to_pydatetime()
        except Exception:
            return None

    now = datetime.now(timezone.utc)
    filtered = []
    for e in events:
        dt = parse_dt(e.get("datetime"))
        if not dt:
            continue
        if dt >= now - timedelta(hours=1):
            e["datetime"] = dt.isoformat()
            filtered.append(e)

    # Sort and return top 10
    filtered.sort(key=lambda x: x["datetime"])
    return filtered[:10]


def rule_based_summary(signals: Dict[str, Any], bias: Dict[str, Any], calendar: List[Dict[str, Any]]) -> Dict[str, Any]:
    levels = signals["levels"]
    ch5 = signals["chg_5d"]
    ch20 = signals["chg_20d"]

    def fmt_pct(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x:+.2f}%"

    def fmt_bp(x: Optional[float]) -> str:
        # yields changes are absolute; display as basis points
        return "n/a" if x is None else f"{x*100:+.0f} bp"

    # Drivers (max 3) per asset
    us_drivers = []
    us_drivers.append(f"S&P 500 momentum (20d): {fmt_pct(ch20['sp500'])}")
    us_drivers.append(f"Volatility (VIX 20d): {fmt_pct(ch20['vix'])}")
    us_drivers.append(f"10Y yields (20d): {fmt_bp(ch20['dgs10'])}")

    xau_drivers = []
    xau_drivers.append(f"Real yields (10Y TIPS 20d): {fmt_bp(ch20['dfii10'])}")
    xau_drivers.append(f"USD (broad index 20d): {fmt_pct(ch20['usd_broad'])}")
    xau_drivers.append(f"Risk tone (VIX 20d): {fmt_pct(ch20['vix'])}")

    # What changed (delta style)
    changes = [
        f"USD: {bias['deltas']['usd']} (5d {fmt_pct(ch5['usd_broad'])})",
        f"Real yields: {bias['deltas']['real_yields']} (5d {fmt_bp(ch5['dfii10'])})",
        f"VIX: {bias['deltas']['vix']} (5d {fmt_pct(ch5['vix'])})",
    ]

    # Actionable guidance (context)
    us_action = []
    if bias["us500"]["bias"] == "bullish":
        us_action += ["Prioritise long setups on pullbacks; avoid chasing into major data."]
    elif bias["us500"]["bias"] == "bearish":
        us_action += ["Size down longs; shorts only with clear risk-off catalyst + clean structure."]
    else:
        us_action += ["Mixed tape: be selective; prefer A+ setups or stand aside."]

    xau_action = []
    if bias["xauusd"]["bias"] == "bullish":
        xau_action += ["Prefer buy-the-dip setups when real yields are falling; watch USD bounce risk."]
    elif bias["xauusd"]["bias"] == "bearish":
        xau_action += ["Fade rallies into resistance if USD/real yields firm; avoid knife-catching."]
    else:
        xau_action += ["Chop risk: wait for post-event direction; trade reactions, not expectations."]

    # Upcoming events: keep only very high impact keywords
    hi_kw = ("CPI", "PCE", "NFP", "FOMC", "Powell", "Fed", "GDP", "Retail Sales", "ISM", "Jobs")
    upcoming = [e for e in calendar if any((e.get("event") or "").upper().find(k.upper()) >= 0 for k in hi_kw)]
    upcoming = upcoming[:6]

    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "levels": {
            "sp500": levels["sp500"],
            "gold": levels["gold"],
            "usd_broad": levels["usd_broad"],
            "dgs10": levels["dgs10"],
            "dfii10": levels["dfii10"],
            "vix": levels["vix"],
        },
        "bias": {
            "us500": bias["us500"],
            "xauusd": bias["xauusd"],
        },
        "drivers": {
            "us500": us_drivers[:3],
            "xauusd": xau_drivers[:3],
        },
        "what_changed": changes,
        "upcoming_risk": upcoming,
        "actions": {
            "us500": us_action,
            "xauusd": xau_action,
        },
        "raw": {
            "chg_5d": signals["chg_5d"],
            "chg_20d": signals["chg_20d"],
        },
    }


def ai_polish(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Optionally rewrites driver/action bullets into cleaner trader language.
    Keeps the structure identical.
    """
    if not OPENAI_API_KEY:
        return snapshot

    # Minimal OpenAI REST call (user runs this; requires internet)
    url = "https://api.openai.com/v1/chat/completions"
    sys = (
        "You are a pro macro-to-trading analyst. Rewrite bullets into crisp, trader-first snapshots. "
        "Rules: Keep each bullet <= 18 words. No forecasts. No hype. No extra sections. "
        "Only rewrite existing bullet text; do not add new bullets or change meaning."
    )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(snapshot, ensure_ascii=False)},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        out = r.json()
        content = out["choices"][0]["message"]["content"]
        rewritten = json.loads(content)
        # Basic safety: ensure key sections exist
        for k in ("drivers", "actions", "what_changed"):
            if k not in rewritten:
                return snapshot
        return rewritten
    except Exception:
        return snapshot


def main() -> None:
    series = {name: fred_observations(sid) for name, sid in FRED_SERIES.items()}
    signals = compute_signals(series)
    bias = score_bias(signals)

    start = datetime.now(timezone.utc)
    end = start + timedelta(days=7)
    calendar = fetch_calendar(start, end)

    snapshot = rule_based_summary(signals, bias, calendar)
    snapshot = ai_polish(snapshot)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
