#!/usr/bin/env python3
"""
Simulate **input drift** for the Streamlit drift dashboard (same logic as `app.py`).

Use this to populate `input_monitoring_log.csv` with either:
  - **baseline** — sample from the same population as `raw_data.csv` (traffic light usually **green**).
  - **drift** — deliberately shifted customer / product / order-value mix (traffic light often **yellow** or **red**).

Run from the `webapp/` directory (needs `raw_data.csv` from `evaluate_models.py`):

  cd assignment2/webapp
  python simulate_drift.py --scenario baseline
  python simulate_drift.py --scenario drift

Copy the printed **Report snippet** into your assignment report. Open the Drift tab in Streamlit to match the tables.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

# Keep in sync with webapp/app.py
LOG_COLUMNS: list[str] = [
    'CustomerID',
    'ProductID',
    'SalesOrderID',
    'OrderDate',
    'LineTotal',
    'kind',
    'score',
    'threshold',
    'predicted_purchase',
    'cold_start_user',
    'ts_utc',
]
NUMERIC_DRIFT = ['CustomerID', 'ProductID', 'SalesOrderID', 'LineTotal']


def _traffic_light(ks_rows: list[dict], chi_rows: list[dict]) -> tuple[str, float, int, int]:
    drift_count = sum(r['drift'] for r in ks_rows) + sum(r['drift'] for r in chi_rows)
    total = len(ks_rows) + len(chi_rows)
    if total == 0:
        return 'n/a (no tests ran)', float('nan'), drift_count, total
    ratio = drift_count / total
    if ratio == 0:
        return 'green', ratio, drift_count, total
    if ratio < 0.34:
        return 'yellow', ratio, drift_count, total
    return 'red', ratio, drift_count, total


def run_ks(train_df: pd.DataFrame, prod_df: pd.DataFrame) -> list[dict]:
    ks_rows: list[dict] = []
    for col in NUMERIC_DRIFT:
        if col not in train_df.columns or col not in prod_df.columns:
            continue
        a = pd.to_numeric(train_df[col], errors='coerce').dropna()
        b = pd.to_numeric(prod_df[col], errors='coerce').dropna()
        if len(a) < 20 or len(b) < 5:
            continue
        stat, p = stats.ks_2samp(a, b, method='auto')
        ks_rows.append({'feature': col, 'KS_statistic': stat, 'p_value': p, 'drift': p < 0.05})
    return ks_rows


def build_baseline(train_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Same broad distribution as reference (low expected KS drift)."""
    r = min(n, len(train_df))
    sample = train_df.sample(r, random_state=seed)
    return _to_log_frame(sample)


def build_shifted(train_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Deliberate **covariate shift**:
    - restrict to high **CustomerID** (upper tail),
    - restrict to high **ProductID** (upper tail),
    - inflate **LineTotal** (order mix looks more “premium” than training).
    """
    t = train_df.copy()
    t['CustomerID'] = pd.to_numeric(t['CustomerID'], errors='coerce')
    t['ProductID'] = pd.to_numeric(t['ProductID'], errors='coerce')
    t['LineTotal'] = pd.to_numeric(t['LineTotal'], errors='coerce')
    t = t.dropna(subset=['CustomerID', 'ProductID', 'LineTotal'])
    c_hi = t['CustomerID'].quantile(0.88)
    p_hi = t['ProductID'].quantile(0.85)
    shifted = t[(t['CustomerID'] >= c_hi) & (t['ProductID'] >= p_hi)].copy()
    if len(shifted) < max(30, n // 2):
        # fallback: looser filter so we always have enough rows
        shifted = t[t['CustomerID'] >= t['CustomerID'].quantile(0.80)].copy()
    r = min(n, len(shifted))
    sample = shifted.sample(r, random_state=seed)
    sample = sample.copy()
    sample['LineTotal'] = sample['LineTotal'] * 2.2
    return _to_log_frame(sample)


def _to_log_frame(sample: pd.DataFrame) -> pd.DataFrame:
    out = sample.reindex(columns=LOG_COLUMNS, fill_value=np.nan)
    for c in ['kind', 'score', 'threshold', 'predicted_purchase', 'cold_start_user', 'ts_utc']:
        if c not in out.columns or out[c].isna().all():
            out[c] = np.nan
    out['ts_utc'] = datetime.now(timezone.utc).isoformat()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Simulate input drift for the assignment dashboard.')
    parser.add_argument(
        '--scenario',
        choices=('baseline', 'drift'),
        default='drift',
        help='baseline: sample from full data; drift: shifted segment + inflated LineTotal',
    )
    parser.add_argument('--n', type=int, default=500, help='Rows to write to the production log.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--output',
        default='input_monitoring_log.csv',
        help='Output path (relative to webapp/ if not absolute).',
    )
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base, 'raw_data.csv')
    out_path = args.output if os.path.isabs(args.output) else os.path.join(base, args.output)

    if not os.path.isfile(train_path):
        raise SystemExit(
            f'Missing {train_path}. Run `python evaluate_models.py` from `src/` to copy raw_data into webapp/.'
        )

    train_df = pd.read_csv(train_path)
    if args.scenario == 'baseline':
        prod_df = build_baseline(train_df, args.n, args.seed)
        desc = (
            'Random sample from **full** `raw_data.csv` (same population as the reference). '
            'KS tests should usually **not** reject at α=0.05.'
        )
    else:
        prod_df = build_shifted(train_df, args.n, args.seed)
        desc = (
            '**Shifted** traffic: upper-tail **CustomerID** and **ProductID**, **LineTotal × 2.2** '
            '(simulates e.g. different customer segment + price mix). Expect **input drift** on several features.'
        )

    prod_df.to_csv(out_path, index=False)
    ks_rows = run_ks(train_df, prod_df)
    light, ratio, dcount, ntests = _traffic_light(ks_rows, [])

    print()
    print('=' * 72)
    print('INPUT DRIFT SIMULATION (for report)')
    print('=' * 72)
    print(f"Scenario: **{args.scenario}** — {desc}")
    print(f"Wrote: `{out_path}` ({len(prod_df)} rows)")
    print(f"Reference rows: {len(train_df):,} (`raw_data.csv`)")
    print()
    if not ks_rows:
        print('No KS tests ran (check column overlap and sample sizes).')
    else:
        tbl = pd.DataFrame(ks_rows)
        print(tbl.to_string(index=False))
        print()
    print(f"Drift flags (p < 0.05): {dcount} / {ntests} tests")
    print(f"Traffic light (same rule as app): **{light}** (ratio = {ratio:.2f})")
    print()
    print('--- Report snippet (markdown) ---')
    print()
    print(
        f"We simulated **{args.scenario}** production traffic by writing `{os.path.basename(out_path)}` "
        f'({len(prod_df)} rows). '
        f'Using the same Kolmogorov–Smirnov tests as the dashboard (α = 0.05), '
        f'{dcount} of {ntests} numeric features showed **statistical input drift**. '
        f"The automated traffic light was **{light}** (fraction of tests with drift ≈ {ratio:.2f})."
    )
    print()
    print('Re-open the Streamlit **Drift dashboard** tab to see the matching table and traffic light.')
    print()


if __name__ == '__main__':
    main()
