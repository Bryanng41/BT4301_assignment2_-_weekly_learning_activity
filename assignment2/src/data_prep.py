"""Line-level interactions, temporal train/test per customer, and recsys id maps / CSR."""
from __future__ import annotations

import json
import os
import pickle
from typing import Any

import mysql.connector
import numpy as np
import pandas as pd
from scipy import sparse

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "database": "datawarehouse",
}

INTERACTION_QUERY = """
SELECT
    s.CustomerID,
    s.ProductID,
    s.SalesOrderID,
    t.OrderDate,
    COALESCE(s.LineTotal, 1.0) AS LineTotal
FROM sales s
    JOIN product p ON s.ProductID = p.ProductID
    JOIN customer c ON s.CustomerID = c.CustomerID
    JOIN salesordertime t ON s.SalesOrderID = t.SalesOrderID
WHERE s.CustomerID IS NOT NULL AND s.ProductID IS NOT NULL
"""


def load_interactions() -> pd.DataFrame:
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql(INTERACTION_QUERY, conn)
    conn.close()
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    df = df.dropna(subset=["CustomerID", "ProductID", "OrderDate"])
    df["CustomerID"] = df["CustomerID"].astype(np.int64)
    df["ProductID"] = df["ProductID"].astype(np.int64)
    return df


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per customer: sort by OrderDate (then SalesOrderID); last ceil(n * test_size)
    interactions go to test. Customers with <2 interactions are train-only.
    """
    _ = random_state  # reserved for future stratified variants
    df = df.sort_values(
        ["CustomerID", "OrderDate", "SalesOrderID"],
        kind="mergesort",
    ).reset_index(drop=True)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, g in df.groupby("CustomerID", sort=False):
        n = len(g)
        if n < 2:
            train_parts.append(g)
            continue
        n_test = max(1, int(np.ceil(n * test_size)))
        train_parts.append(g.iloc[:-n_test])
        test_parts.append(g.iloc[-n_test:])
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0]
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0]
    return train_df, test_df


def build_id_maps(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    product_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Stable contiguous indices for all customers and products seen in train or test."""
    customers = pd.unique(
        pd.concat(
            [train_df["CustomerID"], test_df["CustomerID"]], ignore_index=True
        )
    )
    products = pd.unique(
        pd.concat(
            [train_df["ProductID"], test_df["ProductID"]], ignore_index=True
        )
    )
    customers = np.sort(customers)
    products = np.sort(products)
    customer_id_to_idx = {int(c): i for i, c in enumerate(customers)}
    product_id_to_idx = {int(p): j for j, p in enumerate(products)}
    idx_to_customer_id = [int(c) for c in customers]
    idx_to_product_id = [int(p) for p in products]
    meta: dict[str, Any] = {
        "task": "recommendation",
        "customer_id_to_idx": customer_id_to_idx,
        "product_id_to_idx": product_id_to_idx,
        "idx_to_customer_id": idx_to_customer_id,
        "idx_to_product_id": idx_to_product_id,
        "n_users": len(customers),
        "n_items": len(products),
        "product_id_to_name": {},
    }
    if product_names:
        for pid, name in product_names.items():
            meta["product_id_to_name"][str(int(pid))] = str(name)
    return meta


def load_product_names() -> dict[int, str]:
    q = """
    SELECT ProductID, COALESCE(ProductModelName, CAST(ProductID AS CHAR)) AS DisplayName
    FROM product
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    dfp = pd.read_sql(q, conn)
    conn.close()
    dfp = dfp.dropna(subset=["ProductID"])
    dfp["ProductID"] = dfp["ProductID"].astype(np.int64)
    return dict(zip(dfp["ProductID"], dfp["DisplayName"].astype(str)))


def build_train_csr(
    train_df: pd.DataFrame,
    meta: dict[str, Any],
    binary: bool = True,
) -> sparse.csr_matrix:
    """User × item CSR from train interactions (aggregated duplicate u,i)."""
    cui = meta["customer_id_to_idx"]
    pii = meta["product_id_to_idx"]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for _, r in train_df.iterrows():
        u = int(r["CustomerID"])
        i = int(r["ProductID"])
        if u not in cui or i not in pii:
            continue
        rows.append(cui[u])
        cols.append(pii[i])
        data.append(1.0 if binary else float(r.get("LineTotal", 1.0)))
    n_users = meta["n_users"]
    n_items = meta["n_items"]
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    mat.sum_duplicates()
    if binary:
        mat.data = np.ones_like(mat.data)
    return mat


def test_pairs_from_df(
    test_df: pd.DataFrame,
    meta: dict[str, Any],
    train_csr: sparse.csr_matrix,
) -> list[tuple[int, int]]:
    """
    List of (CustomerID, ProductID) for evaluation. Drops pairs whose product
    never appears in train (cold item) so scores are defined from train stats.
    """
    train_items = set()
    train_csr = train_csr.tocoo()
    for c, r in zip(train_csr.col, train_csr.row):
        train_items.add(int(meta["idx_to_product_id"][c]))
    pairs: list[tuple[int, int]] = []
    for _, r in test_df.iterrows():
        u = int(r["CustomerID"])
        p = int(r["ProductID"])
        if p not in train_items:
            continue
        if u not in meta["customer_id_to_idx"]:
            continue
        pairs.append((u, p))
    return pairs


def train_user_seen_items(train_csr: sparse.csr_matrix) -> list[set[int]]:
    """For each user row index, set of item column indices with a train interaction."""
    out = [set() for _ in range(train_csr.shape[0])]
    coo = train_csr.tocoo()
    for u, j in zip(coo.row, coo.col):
        out[int(u)].add(int(j))
    return out


def write_recommender_metadata_pickle(path: str, meta: dict[str, Any]) -> None:
    """Persist id maps, test pairs, CSR helpers, etc. (from `prepare_interaction_split`) as a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(meta, f)


def read_recommender_metadata_pickle(path: str) -> dict[str, Any]:
    """Load a `recsys_meta.pkl` written by `write_recommender_metadata_pickle`."""
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_interaction_split(
    df: pd.DataFrame | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], sparse.csr_matrix]:
    """
    Load (if df is None), split, build id maps and product names, train CSR,
    and attach test_pairs + train_user_items to meta.
    """
    if df is None:
        df = load_interactions()
    train_df, test_df = temporal_train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    try:
        names = load_product_names()
    except Exception:
        names = {
            int(p): str(p)
            for p in pd.unique(
                pd.concat(
                    [train_df["ProductID"], test_df["ProductID"]], ignore_index=True
                )
            )
        }
    meta = build_id_maps(train_df, test_df, product_names=names)
    meta["test_size"] = float(test_size)
    meta["random_state"] = int(random_state)
    train_csr = build_train_csr(train_df, meta, binary=True)
    meta["train_user_items"] = train_user_seen_items(train_csr)
    meta["test_pairs"] = test_pairs_from_df(test_df, meta, train_csr)
    # popularity for cold-start (product idx -> count on train)
    meta["item_popularity"] = np.asarray(train_csr.sum(axis=0)).ravel().tolist()
    return train_df, test_df, meta, train_csr


def write_train_interaction_stats_json(train_df: pd.DataFrame, output_path: str) -> None:
    """Write per-column numeric summaries of the train interaction frame to JSON (drift / reporting)."""
    stats: dict[str, Any] = {}
    for col in train_df.select_dtypes(include=[np.number]).columns:
        s = train_df[col].dropna()
        if len(s) == 0:
            continue
        stats[str(col)] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
        }
    stats["n_rows"] = int(len(train_df))
    stats["n_customers"] = int(train_df["CustomerID"].nunique())
    stats["n_products"] = int(train_df["ProductID"].nunique())
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)

    print("Loading interactions from warehouse...")
    train_df, test_df, meta, train_csr = prepare_interaction_split()
    print(f"  Train lines: {len(train_df)}, test lines: {len(test_df)}")
    print(f"  Users: {meta['n_users']}, items: {meta['n_items']}, train CSR nnz: {train_csr.nnz}")
    print(f"  Eval test pairs (cold-item filtered): {len(meta['test_pairs'])}")

    train_df.to_csv("../data/interactions_train.csv", index=False)
    test_df.to_csv("../data/interactions_test.csv", index=False)
    pd.concat([train_df, test_df], ignore_index=True).to_csv(
        "../data/raw_data.csv", index=False
    )
    write_recommender_metadata_pickle("../data/recsys_meta.pkl", meta)
    write_train_interaction_stats_json(train_df, "../data/training_stats.json")

    with open("../data/feature_names.json", "w") as f:
        json.dump(["CustomerID", "ProductID", "OrderDate"], f, indent=2)

    print("Done. Wrote ../data/interactions_*.csv, raw_data.csv, recsys_meta.pkl, training_stats.json")
