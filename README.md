# Fundamentals Snapshot Dashboard (US500 + XAUUSD)

Auto-updating, trader-first fundamentals dashboard for:
- **US500** (S&P 500 proxy)
- **XAUUSD** (Gold)

## Outputs
- `public/latest.json` — the dashboard payload
- `public/index.html` — static dashboard UI

## Data sources
- **FRED** (St. Louis Fed) time series (prices + macro proxies):
  - S&P 500: `SP500`
  - Gold LBMA AM fix: `GOLDAMGBD228NLBM`
  - 10Y nominal: `DGS10`
  - 10Y TIPS (real yield): `DFII10`
  - VIX: `VIXCLS`
  - Broad USD index: `DTWEXBGS`
- Optional economic calendar:
  - Trading Economics Calendar API (`calendar.aspx`) or FMP Economic Calendar API.

## Quick start (local)
1) Install Python deps:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
2) Create `.env`:
```bash
cp .env.example .env
# add FRED_API_KEY (free), optionally TE_API_KEY or FMP_API_KEY
# add OPENAI_API_KEY if you want AI-written snapshots
```
3) Build payload:
```bash
python build_latest.py
```
4) Open dashboard:
```bash
python -m http.server 8000 --directory public
# visit http://localhost:8000
```

## Auto-update options
- **GitHub Actions**: run `python build_latest.py` on a schedule and publish `public/` via GitHub Pages.
- **Local cron**: run `python build_latest.py` every hour.

## Notes
- If OPENAI_API_KEY is not set, the project falls back to deterministic rule-based summaries.
- The bias logic is designed for *context*, not signals.
