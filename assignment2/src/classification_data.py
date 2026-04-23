from __future__ import annotations

import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

# These features must be the same as the ones used to train the models
CATEGORICAL_FEATURES = [
    "City",
    "StateProvinceName",
    "CountryName",
    "ProductCategoryName",
    "ProductSubCategoryName",
    "ProductModelName",
    "Color",
]
NUMERIC_FEATURES = [
    "CustomerID",
    "ProductID",
    "ListPrice",
    "order_year",
    "order_month",
    "order_dow",
    "user_train_lines_total",
    "product_train_line_count",
    "user_order_seq",
]

CLASSIFICATION_ENRICH_QUERY = """
SELECT
    s.CustomerID,
    s.ProductID,
    s.SalesOrderID,
    t.OrderDate,
    COALESCE(s.LineTotal, 1.0) AS LineTotal,
    p.ProductCategoryName,
    p.ProductSubCategoryName,
    p.ProductModelName,
    p.Color,
    p.ListPrice,
    c.City,
    c.StateProvinceName,
    c.CountryName
FROM sales s
    JOIN product p ON s.ProductID = p.ProductID
    JOIN customer c ON s.CustomerID = c.CustomerID
    JOIN salesordertime t ON s.SalesOrderID = t.SalesOrderID
WHERE s.CustomerID IS NOT NULL AND s.ProductID IS NOT NULL
"""


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    t = pd.to_datetime(d["OrderDate"], errors="coerce")
    d["order_year"] = t.dt.year.fillna(2000).astype(int)
    d["order_month"] = t.dt.month.fillna(1).astype(int)
    d["order_dow"] = t.dt.dayofweek.fillna(0).astype(int)
    return d


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in CATEGORICAL_FEATURES + ["ListPrice"]:
        if c not in df.columns:
            df[c] = np.nan
    return df


def load_enriched_interactions() -> pd.DataFrame:
    """
    Line-level data with product + customer attributes.
    Tries MySQL; falls back to raw_data + empty enrich if DB unavailable.
    """
    from data_prep import DB_CONFIG, load_interactions

    import mysql.connector

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        try:
            df = pd.read_sql(CLASSIFICATION_ENRICH_QUERY, conn)
        finally:
            conn.close()
    except Exception as e:
        # Fallback: CSV from data pipeline (no MySQL)
        print(f"classification_data: MySQL load failed ({e!r}). Trying data/raw_data.csv fallback.")
        raw = os.path.join(os.path.dirname(__file__), "..", "data", "raw_data.csv")
        if os.path.isfile(raw):
            df = pd.read_csv(raw, parse_dates=["OrderDate"])
            for c in [
                "ProductCategoryName",
                "ProductSubCategoryName",
                "ProductModelName",
                "Color",
                "ListPrice",
                "City",
                "StateProvinceName",
                "CountryName",
                "SalesOrderID",
            ]:
                if c not in df.columns:
                    df[c] = np.nan
            if "LineTotal" not in df.columns:
                df["LineTotal"] = 1.0
            if "SalesOrderID" in df.columns:
                df["SalesOrderID"] = pd.to_numeric(df["SalesOrderID"], errors="coerce").fillna(-1).astype(np.int64)
        else:
            print("No raw_data.csv; using load_interactions() (requires MySQL).")
            df = load_interactions()
            for c in [
                "ProductCategoryName",
                "ProductSubCategoryName",
                "ProductModelName",
                "Color",
                "ListPrice",
                "City",
                "StateProvinceName",
                "CountryName",
            ]:
                if c not in df.columns:
                    df[c] = np.nan
            if "LineTotal" not in df.columns:
                df["LineTotal"] = 1.0
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    df = df.dropna(subset=["CustomerID", "ProductID", "OrderDate"])
    df["CustomerID"] = df["CustomerID"].astype(np.int64)
    df["ProductID"] = df["ProductID"].astype(np.int64)
    if "ListPrice" in df.columns:
        df["ListPrice"] = pd.to_numeric(df["ListPrice"], errors="coerce")
    return _ensure_cols(_add_time_features(df))


def _train_only_aggregates(
    train_df: pd.DataFrame, split_df: pd.DataFrame, *, is_train_positive: bool
) -> pd.DataFrame:
    """Add user_train_lines_total, product_train_line_count, user_order_seq on split_df."""
    t = train_df.sort_values(
        ["CustomerID", "OrderDate", "SalesOrderID"],
        kind="mergesort",
    ).copy()
    t["user_order_seq"] = t.groupby("CustomerID", sort=False).cumcount() + 1
    user_tot = t.groupby("CustomerID", sort=False).size().rename("user_train_lines_total")
    prod_cnt = t.groupby("ProductID", sort=False).size().rename("product_train_line_count")
    out = split_df.copy()
    out = out.merge(user_tot, on="CustomerID", how="left")
    out = out.merge(prod_cnt, on="ProductID", how="left")
    if is_train_positive and "SalesOrderID" in out.columns:
        m = t[
            [
                "CustomerID",
                "ProductID",
                "OrderDate",
                "SalesOrderID",
                "user_order_seq",
            ]
        ].copy()
        out = out.merge(
            m,
            on=["CustomerID", "ProductID", "OrderDate", "SalesOrderID"],
            how="left",
        )
    else:
        # synthetic negatives: use last line index in train for that user
        last_seq = t.groupby("CustomerID", sort=False)["user_order_seq"].max()
        out["user_order_seq"] = out["CustomerID"].map(last_seq).fillna(0.0)
    for c in ("user_train_lines_total", "product_train_line_count", "user_order_seq"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _apply_train_aggregates_for_test(
    train_df: pd.DataFrame, test_pos: pd.DataFrame
) -> pd.DataFrame:
    """Test rows: user/product counts from train only; test order index after last train line."""
    user_tot = train_df.groupby("CustomerID", sort=False).size().rename("user_train_lines_total")
    prod_cnt = train_df.groupby("ProductID", sort=False).size().rename("product_train_line_count")
    trs = train_df.sort_values(
        ["CustomerID", "OrderDate", "SalesOrderID"],
        kind="mergesort",
    )
    trs = trs.copy()
    trs["uord"] = trs.groupby("CustomerID", sort=False).cumcount() + 1
    last_tr = trs.groupby("CustomerID", sort=False)["uord"].max()
    out = test_pos.copy()
    out = out.merge(user_tot, on="CustomerID", how="left")
    out = out.merge(prod_cnt, on="ProductID", how="left")
    te = test_pos.sort_values(
        ["CustomerID", "OrderDate", "SalesOrderID"],
        kind="mergesort",
    ).copy()
    te["v"] = te.groupby("CustomerID", sort=False).cumcount() + 1
    base = te["CustomerID"].map(last_tr).fillna(0.0)
    te["user_order_seq"] = base + te["v"]
    key = [c for c in ["CustomerID", "ProductID", "OrderDate", "SalesOrderID"] if c in out.columns and c in te.columns]
    out = out.drop(columns=[c for c in ["user_order_seq"] if c in out.columns], errors="ignore")
    out = out.merge(te[key + ["user_order_seq"]], on=key, how="left")
    for c in ("user_train_lines_total", "product_train_line_count", "user_order_seq"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _sample_negatives(
    users: list[int],
    user_forbidden: dict[int, set[int]],
    product_pool: np.ndarray,
    n_neg: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """
    For each user, sample n_neg products from product_pool
    that are not in user_forbidden[user].
    """
    rows: list[tuple[int, int]] = []
    pool_set = set(int(x) for x in product_pool)
    for u in users:
        forbid = user_forbidden.get(int(u), set())
        allowed = [p for p in pool_set if p not in forbid]
        if not allowed:
            continue
        take = min(n_neg, len(allowed))
        pick = rng.choice(allowed, size=take, replace=(take < n_neg)) if take else []
        for p in pick:
            rows.append((int(u), int(p)))
        # if take < n_neg, pad with resample
        if take < n_neg and allowed:
            extra = n_neg - take
            more = rng.choice(allowed, size=extra, replace=True)
            for p in more:
                rows.append((int(u), int(p)))
    return rows

def build_train_test_event_tables(
    df: pd.DataFrame | None = None,
    *,
    test_size: float = 0.2,
    n_neg: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:

    from data_prep import temporal_train_test_split

    if df is None:
        df = load_enriched_interactions()
    else:
        df = _ensure_cols(_add_time_features(df.copy()))
    train_df, test_df = temporal_train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    pool_train = np.sort(train_df["ProductID"].unique())
    if pool_train.size == 0:
        raise ValueError("No products in train split.")

    train_user_purch: dict[int, set[int]] = (
        train_df.groupby("CustomerID")["ProductID"].apply(lambda s: set(s.astype(int))).to_dict()
    )
    # max OrderDate in train per user (for neg rows)
    train_max_date = train_df.groupby("CustomerID", sort=False)["OrderDate"].max().to_dict()

    rng = np.random.default_rng(random_state)

    # ---- Train events ----
    pos_tr = train_df.copy()
    pos_tr["y"] = 1
    pos_tr = _train_only_aggregates(train_df, pos_tr, is_train_positive=True)

    neg_u = list(train_user_purch.keys())
    neg_pairs = _sample_negatives(
        neg_u,
        train_user_purch,
        pool_train,
        n_neg,
        rng,
    )
    if not neg_pairs:
        raise ValueError("No negative train rows; check data.")

    neg_rows: list[dict[str, Any]] = []
    for u, p in neg_pairs:
        odt = train_max_date.get(u, train_df["OrderDate"].min())
        neg_rows.append(
            {
                "CustomerID": u,
                "ProductID": p,
                "SalesOrderID": -1,
                "OrderDate": odt,
                "y": 0,
            }
        )
    neg_df = pd.DataFrame(neg_rows)
    # merge product + customer + time for negatives (need lookup from full df)
    pcols = [
        c
        for c in [
            "ProductCategoryName",
            "ProductSubCategoryName",
            "ProductModelName",
            "Color",
            "ListPrice",
        ]
        if c in df.columns
    ]
    dim_p = df.drop_duplicates("ProductID").set_index("ProductID")[pcols]
    dim_c = df.drop_duplicates("CustomerID").set_index("CustomerID")[
        [c for c in CATEGORICAL_FEATURES if c in ["City", "StateProvinceName", "CountryName"]]
    ]
    neg_df = neg_df.join(dim_p, on="ProductID")
    neg_df = neg_df.join(dim_c, on="CustomerID")
    neg_df = _add_time_features(neg_df)
    neg_df = _train_only_aggregates(train_df, neg_df, is_train_positive=False)

    train_events = pd.concat([pos_tr, neg_df], ignore_index=True, sort=False)
    train_events = train_events.fillna({"LineTotal": 1.0})
    for c in CATEGORICAL_FEATURES:
        if c in train_events.columns:
            train_events[c] = train_events[c].fillna("unknown").astype(str)
        else:
            train_events[c] = "unknown"
    for c in NUMERIC_FEATURES + ["y"]:
        if c not in train_events.columns:
            train_events[c] = 0.0
    # ---- Test events ----
    test_user_purch = (
        test_df.groupby("CustomerID")["ProductID"]
        .apply(lambda s: set(s.astype(int)))
        .to_dict()
    )
    test_users = [u for u in test_user_purch if u in test_df["CustomerID"].values]
    pos_te = test_df.copy()
    pos_te["y"] = 1
    pos_te = _apply_train_aggregates_for_test(train_df, pos_te)

    forbidden: dict[int, set[int]] = {}
    for u in set(train_user_purch) | set(test_user_purch):
        s = set(train_user_purch.get(u, set())) | set(test_user_purch.get(u, set()))
        forbidden[int(u)] = s

    neg_pairs_te = _sample_negatives(
        list(test_user_purch.keys()),
        forbidden,
        pool_train,
        n_neg,
        rng,
    )
    neg_te: list[dict[str, Any]] = []
    first_test = test_df.sort_values(["CustomerID", "OrderDate", "SalesOrderID"]).groupby("CustomerID")["OrderDate"].min()
    for u, p in neg_pairs_te:
        odt = first_test.get(u, train_max_date.get(u, test_df["OrderDate"].min()))
        if pd.isna(odt):
            odt = train_max_date.get(u, test_df["OrderDate"].min())
        neg_te.append(
            {
                "CustomerID": u,
                "ProductID": p,
                "SalesOrderID": -1,
                "OrderDate": odt,
                "y": 0,
            }
        )
    neg_te_df = pd.DataFrame(neg_te)
    neg_te_df = neg_te_df.join(dim_p, on="ProductID")
    neg_te_df = neg_te_df.join(dim_c, on="CustomerID")
    neg_te_df = _add_time_features(neg_te_df)
    neg_te_df = _apply_train_aggregates_for_test(train_df, neg_te_df)

    test_events = pd.concat([pos_te, neg_te_df], ignore_index=True, sort=False)
    test_events = test_events.fillna({"LineTotal": 1.0})
    for c in CATEGORICAL_FEATURES:
        if c in test_events.columns:
            test_events[c] = test_events[c].fillna("unknown").astype(str)
        else:
            test_events[c] = "unknown"
    for c in NUMERIC_FEATURES + ["y"]:
        if c not in test_events.columns:
            test_events[c] = 0.0

    aux = {
        "n_neg": n_neg,
        "test_size": test_size,
        "random_state": random_state,
        "product_pool_n": int(len(pool_train)),
    }
    return train_events, test_events, aux

# This function saves the product and customer feature rows for the API inference
def save_product_customer_dims(df: pd.DataFrame, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    pcols = [
        "ProductID",
        "ProductCategoryName",
        "ProductSubCategoryName",
        "ProductModelName",
        "Color",
        "ListPrice",
    ]
    ccols = ["CustomerID", "City", "StateProvinceName", "CountryName"]
    p = (
        df[pcols].drop_duplicates("ProductID")
        if all(c in df.columns for c in pcols)
        else pd.DataFrame()
    )
    c = (
        df[ccols].drop_duplicates("CustomerID")
        if all(c in df.columns for c in ccols)
        else pd.DataFrame()
    )
    path = os.path.join(out_dir, "dim_tables.pkl")
    with open(path, "wb") as f:
        pickle.dump({"product_dim": p, "customer_dim": c}, f)
    return path


def save_event_csvs(
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    out_dir: str,
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    tpath = os.path.join(out_dir, "train_events.csv")
    ppath = os.path.join(out_dir, "test_events.csv")
    train_events.to_csv(tpath, index=False)
    test_events.to_csv(ppath, index=False)
    return tpath, ppath


def load_or_build_event_tables(
    out_dir: str,
    *,
    n_neg: int = 5,
    random_state: int = 42,
    always_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reuse `train_events.csv` / `test_events.csv` when present (fast reruns);
    otherwise build from `load_enriched_interactions` and write CSVs.
    If ``always_rebuild`` is True, always rebuild and overwrite CSVs (baseline).
    """
    tpath = os.path.join(out_dir, "train_events.csv")
    ppath = os.path.join(out_dir, "test_events.csv")
    if always_rebuild or not (os.path.isfile(tpath) and os.path.isfile(ppath)):
        base = load_enriched_interactions()
        tr, te, _ = build_train_test_event_tables(
            base, n_neg=n_neg, random_state=random_state
        )
        save_event_csvs(tr, te, out_dir)
        return tr, te
    return pd.read_csv(tpath), pd.read_csv(ppath)


def save_clf_feature_config(path: str) -> None:
    with open(path, "w") as f:
        json.dump(
            {"categorical": CATEGORICAL_FEATURES, "numeric": NUMERIC_FEATURES},
            f,
            indent=2,
        )
