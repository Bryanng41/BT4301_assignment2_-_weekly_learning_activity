import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy import stats

_BASE = os.path.dirname(os.path.abspath(__file__))

# Single schema: raw_data columns (drift) + API monitoring fields
LOG_PATH = os.path.join(_BASE, 'input_monitoring_log.csv')
LEGACY_LOG_PATH = os.path.join(_BASE, 'production_log.csv')
TRAIN_RAW = os.path.join(_BASE, 'raw_data.csv')
STATS_PATH = os.path.join(_BASE, 'training_stats.json')

# Aligned to raw_data.csv from evaluate_models + prediction metadata
RAW_DRIFT_COLS = ['CustomerID', 'ProductID', 'SalesOrderID', 'OrderDate', 'LineTotal']
MONITORING_EXTRA_COLS = [
    'kind', 'score', 'threshold', 'predicted_purchase', 'cold_start_user', 'ts_utc',
]
LOG_COLUMNS: list[str] = RAW_DRIFT_COLS + MONITORING_EXTRA_COLS

# Drift: interaction-level reference from data_prep (raw_data.csv)
NUMERIC_DRIFT = [
    'CustomerID',
    'ProductID',
    'SalesOrderID',
    'LineTotal',
]

MODEL_TYPE: dict[str, str] = {
    'xgb': 'XGBoost',
    'lgbm': 'LightGBM',
    'lr': 'Logistic regression',
    'logistic_regression': 'Logistic regression',
    'logistic_regression_tuned': 'Logistic regression',
    'log_reg': 'Logistic regression',
    'rf': 'Random forest',
}


def _round2(x) -> str:
    if x is None:
        return '—'
    try:
        return f'{float(x):.2f}'
    except (TypeError, ValueError):
        return '—'


def model_type(kind: str | None) -> str:
    if not kind:
        return '—'
    k = str(kind).lower().strip()
    return MODEL_TYPE.get(k, k.replace('_', ' ').title())


def _historical_purchase_pairs(ref_df: pd.DataFrame) -> set[tuple[int, int]]:
    """(CustomerID, ProductID) with at least one line in reference sales history (`raw_data.csv`)."""
    if 'CustomerID' not in ref_df.columns or 'ProductID' not in ref_df.columns:
        return set()
    t = ref_df[['CustomerID', 'ProductID']].copy()
    t['CustomerID'] = pd.to_numeric(t['CustomerID'], errors='coerce')
    t['ProductID'] = pd.to_numeric(t['ProductID'], errors='coerce')
    t = t.dropna()
    if t.empty:
        return set()
    return {(int(c), int(p)) for c, p in zip(t['CustomerID'], t['ProductID'], strict=True)}


def _ground_truth_label(customer_id, product_id, hist: set[tuple[int, int]]) -> str:
    c = pd.to_numeric(customer_id, errors='coerce')
    p = pd.to_numeric(product_id, errors='coerce')
    if pd.isna(c) or pd.isna(p):
        return '— (missing customer or product id)'
    c_i, p_i = int(c), int(p)
    if (c_i, p_i) in hist:
        return 'Ground truth: this user bought this product before (in reference data).'
    return 'User has not bought this item before.'


def _row_to_log_dataframe(row: dict) -> pd.DataFrame:
    """One row with exactly LOG_COLUMNS (avoids ragged/mixed CSV lines)."""
    full = {c: row.get(c) for c in LOG_COLUMNS}
    return pd.DataFrame([full], columns=LOG_COLUMNS)


def append_log(row: dict) -> None:
    df = _row_to_log_dataframe(row)
    exists = os.path.isfile(LOG_PATH)
    df.to_csv(LOG_PATH, mode='a' if exists else 'w', header=not exists, index=False)


def read_monitoring_log() -> pd.DataFrame | None:
    """Read the monitoring log if present; otherwise None."""
    path = LOG_PATH
    if not os.path.isfile(path) and os.path.isfile(LEGACY_LOG_PATH):
        path = LEGACY_LOG_PATH
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def page_predict():
    st.header('Purchase prediction (customer × product)')
    _default_api = os.environ.get(
        'PREDICT_API_BASE',
        'http://172.17.0.1:5002',
    )
    api_base = st.text_input('REST API base URL', value=_default_api)

    customer_id = st.number_input('Customer ID', value=11000, min_value=1, step=1, format='%d')
    product_id = st.number_input('Product ID', value=771, min_value=1, step=1, format='%d')

    if st.button('Predict purchase'):
        params = {
            'customer_id': int(customer_id),
            'product_id': int(product_id),
        }
        try:
            r = requests.get(f'{api_base.rstrip("/")}/api/purchase', params=params, timeout=60)
        except Exception as e:
            st.error(f'API error: {e}')
            return
        if not r.ok:
            try:
                detail = r.json().get('detail', r.text)
            except Exception:
                detail = r.text
            st.error(f'API error {r.status_code}: {detail}')
            return
        out = r.json()

        if out.get('cold_start_user'):
            st.caption('Cold-start user: not in the training user index; score uses default feature values.')

        kind_raw = (out.get('kind') or '') or ''
        name = (out.get('product_name') or 'Product').strip() or 'Product'
        st.subheader(name)
        pid = int(out.get('product_id', product_id))
        st.caption(f'Customer **{int(customer_id):,}** · Product **{pid:,}**')

        model_name = model_type(kind_raw)
        score_label = 'P(purchase)'
        score_caption = 'Probability of purchase for this customer–product pair (0–1).'
        thr_caption = 'If the score is at or above this cutoff, we predict a purchase.'

        t = out.get('threshold')
        score_s = _round2(out.get('score'))
        thr_s = _round2(t)

        c1, c2, c3 = st.columns(3)
        c1.metric('Model', model_name, help='Which trained model served this request.')
        c2.metric(score_label, score_s, help=score_caption)
        c3.metric('Decision cutoff', thr_s, help=thr_caption)

        pred = out.get('predicted_purchase')
        if pred is True:
            st.success('**Prediction:** this pair is **likely to purchase** (score ≥ cutoff).')
        elif pred is False:
            st.info('**Prediction:** **unlikely to purchase** (score is below the cutoff).')
        else:
            st.warning('**Prediction:** not available (no score or model could not decide).')

        log_row = {
            'CustomerID': int(customer_id),
            'ProductID': int(product_id),
            'SalesOrderID': np.nan,
            'OrderDate': np.nan,
            'LineTotal': np.nan,
            'kind': out.get('kind', ''),
            'score': out.get('score'),
            'threshold': out.get('threshold'),
            'predicted_purchase': out.get('predicted_purchase'),
            'cold_start_user': out.get('cold_start_user', False),
            'ts_utc': datetime.now(timezone.utc).isoformat(),
        }
        append_log(log_row)
        st.caption('Request metadata logged to `input_monitoring_log.csv` for drift monitoring.')


def page_drift():
    st.header('Input drift monitoring')
    if not os.path.isfile(TRAIN_RAW):
        st.warning('Missing `raw_data.csv` (run `evaluate_models.py` after training).')
        return

    train_df = pd.read_csv(TRAIN_RAW)
    has_log = os.path.isfile(LOG_PATH) or os.path.isfile(LEGACY_LOG_PATH)
    if not has_log:
        st.info('No production log yet. Run a purchase prediction from the other tab first.')
        if st.button('Seed demo log (500 random training rows)'):
            sample = train_df.sample(min(500, len(train_df)), random_state=42)
            sample = sample.reindex(columns=LOG_COLUMNS, fill_value=np.nan)
            sample.to_csv(LOG_PATH, index=False)
            st.rerun()
        return

    prod_df = read_monitoring_log()
    if prod_df is None:
        return
    st.write(f'Training reference rows: **{len(train_df):,}** | Production log rows: **{len(prod_df):,}**')

    st.subheader('Ground truth (user × product history)')
    st.caption(
        'For each row in the production log, we check whether the same **customer and product** '
        'appears in **`raw_data.csv`** (historical order lines). That indicates a real prior purchase '
        'in the reference data; API-only requests without a matching line are treated as *no* ground truth for that pair.'
    )
    hist_pairs = _historical_purchase_pairs(train_df)
    st.caption(f'**{len(hist_pairs):,}** distinct (customer, product) pairs in reference data.')

    if 'CustomerID' not in prod_df.columns or 'ProductID' not in prod_df.columns:
        st.warning('Production log has no `CustomerID` / `ProductID` columns; cannot evaluate ground truth.')
    else:
        gt_view = prod_df.copy()
        gt_view['Ground truth'] = gt_view.apply(
            lambda r: _ground_truth_label(r['CustomerID'], r['ProductID'], hist_pairs),
            axis=1,
        )
        n_known = int(gt_view['Ground truth'].str.startswith('Ground truth', na=False).sum())
        n_not_before = int(
            (gt_view['Ground truth'] == 'User has not bought this item before.').sum()
        )
        n_other = int(len(gt_view) - n_known - n_not_before)
        c_a, c_b, c_c = st.columns(3)
        c_a.metric('Log rows (evaluated)', f'{len(gt_view):,}')
        c_b.metric('Prior purchase in reference data', f'{n_known:,}')
        c_c.metric('User has not bought this item before', f'{n_not_before:,}')
        if n_other:
            st.caption(f'**{n_other:,}** row(s) could not be matched (missing ids).')
        show_cols = ['ts_utc', 'CustomerID', 'ProductID', 'Ground truth'] if 'ts_utc' in gt_view.columns else ['CustomerID', 'ProductID', 'Ground truth']
        if 'ts_utc' in gt_view.columns:
            gt_view = gt_view.sort_values('ts_utc', ascending=False)
        st.dataframe(
            gt_view[show_cols].head(500),
            use_container_width=True,
            hide_index=True,
        )
        if len(gt_view) > 500:
            st.caption('Showing the 500 most recent rows (by `ts_utc` when present).')

    ks_rows = []
    for col in NUMERIC_DRIFT:
        if col not in train_df.columns or col not in prod_df.columns:
            continue
        a = pd.to_numeric(train_df[col], errors='coerce').dropna()
        b = pd.to_numeric(prod_df[col], errors='coerce').dropna()
        if len(a) < 20 or len(b) < 5:
            continue
        stat, p = stats.ks_2samp(a, b, method='auto')
        ks_rows.append({'feature': col, 'KS_statistic': stat, 'p_value': p, 'drift': p < 0.05})

    if ks_rows:
        ks_df = pd.DataFrame(ks_rows)
        st.subheader('Numeric features (Kolmogorov–Smirnov)')
        st.dataframe(ks_df)
    else:
        st.caption('No overlapping numeric columns for KS test (check production log schema).')

    drift_count = sum(r['drift'] for r in ks_rows)
    total = len(ks_rows)
    if total == 0:
        return
    ratio = drift_count / total
    if ratio == 0:
        st.success('Traffic light: **green** — no significant drift at α=0.05.')
    elif ratio < 0.34:
        st.warning('Traffic light: **yellow** — some inputs show distribution shift.')
    else:
        st.error('Traffic light: **red** — widespread drift; consider retraining.')

    if os.path.isfile(STATS_PATH):
        with open(STATS_PATH, encoding='utf-8') as f:
            training_stats = json.load(f)
        with st.expander('Training stats (JSON)'):
            st.json(training_stats)


def main():
    st.set_page_config(page_title='BT4301 Assignment 2', layout='wide')
    tab1, tab2 = st.tabs(['Purchase prediction', 'Drift dashboard'])
    with tab1:
        page_predict()
    with tab2:
        page_drift()


if __name__ == '__main__':
    main()
