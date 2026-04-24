"""Cleans and prepares features for each of the four classifiers models."""
from __future__ import annotations

import os
import pickle
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from classification_data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ClfKind = Literal["log_reg", "rf", "xgb", "lgbm"]

# Logistic regression: fixed decision threshold on P(purchase)
LOG_REG_PROBA_THRESHOLD = 0.02


def get_X_only(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in CATEGORICAL_FEATURES:
        if c in d.columns:
            d[c] = d[c].fillna("unknown").astype(str)
    for c in NUMERIC_FEATURES:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    return d[[c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES if c in d.columns]]


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    feat = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna("unknown").astype(str)
    for c in NUMERIC_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    X = df[[c for c in feat if c in df.columns]]
    y = df["y"].values.astype(int)
    return X, y


def build_preprocess() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=25,
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_clf_pipeline(kind: ClfKind) -> tuple[Pipeline, dict[str, Any]]:
    pre = build_preprocess()
    extra: dict[str, Any] = {}
    if kind == "log_reg":
        est = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
            n_jobs=None,
        )
    elif kind == "rf":
        est = RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif kind == "xgb":
        from xgboost import XGBClassifier

        est = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        extra["xgb"] = True
    elif kind == "lgbm":
        from lightgbm import LGBMClassifier

        est = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        extra["lgbm"] = True
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", est)]), extra


def _set_scale_pos_weight_for_grad_boosters(
    pipeline: Pipeline, y: np.ndarray
) -> None:
    """XGBoost and LightGBM use scale_pos_weight for class imbalance (same rule as XGB-only before)."""
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        return
    name = type(clf).__name__
    if name not in ("XGBClassifier", "LGBMClassifier"):
        return
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    spw = (n_neg / n_pos) if n_pos else 1.0
    clf.set_params(scale_pos_weight=float(spw))


def _tune_threshold_f1(
    pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray
) -> float:
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    pos = proba[y == 1]
    if pos.size == 0:
        return 0.1
    t = float(np.clip(np.quantile(pos, 0.05), 0.01, 0.12))
    return t


def fit_clf(
    kind: ClfKind,
    train_df: pd.DataFrame,
) -> tuple[Pipeline, float, dict[str, float]]:
    X, y = get_xy(train_df.copy())
    pipe, _ = make_clf_pipeline(kind)
    _set_scale_pos_weight_for_grad_boosters(pipe, y)
    if kind == "log_reg":
        pipe.fit(X, y)
        threshold = float(LOG_REG_PROBA_THRESHOLD)
    else:
        threshold = _tune_threshold_f1(pipe, X, y)
    metrics: dict[str, float] = {"threshold_tuned": float(threshold)}
    return pipe, threshold, metrics


def predict_metrics(
    pipeline: Pipeline, test_df: pd.DataFrame, *, threshold: float = 0.5
) -> dict[str, Any]:
    X, y = get_xy(test_df.copy())
    proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else None
    if proba is not None:
        pred = (proba >= float(threshold)).astype(int)
    else:
        pred = pipeline.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "y_true": y,
        "y_pred": pred,
        "y_proba": proba,
        "report": classification_report(y, pred, zero_division=0),
        "threshold": float(threshold),
    }


def build_inference_rows_for_user(
    customer_id: int,
    meta: dict[str, Any],
    product_dim: pd.DataFrame,
    customer_dim: pd.DataFrame,
    user_stats: dict[str, Any],
) -> pd.DataFrame:
    """One row per item index for building a single (customer, product) scoring row."""
    n_items = int(meta["n_items"])
    rows = []
    cr = (
        customer_dim[customer_dim["CustomerID"] == int(customer_id)].iloc[0].to_dict()
        if not customer_dim.empty
        and int(customer_id) in set(customer_dim["CustomerID"].values)
        else {c: "unknown" for c in ["City", "StateProvinceName", "CountryName"]}
    )
    ustat = user_stats.get(int(customer_id), {})
    last_dt = ustat.get("last_order", pd.Timestamp("2000-01-01"))
    nlines = float(ustat.get("n_train_lines", 0.0))
    uo = float(ustat.get("user_order_seq", nlines + 1))
    p_counts = ustat.get("product_train_count", {})  # dict product_id -> int

    for j in range(n_items):
        pid = int(meta["idx_to_product_id"][j])
        pr = (
            product_dim[product_dim["ProductID"] == pid].iloc[0].to_dict()
            if not product_dim.empty
            and pid in set(product_dim["ProductID"].values)
            else {}
        )
        t = pd.to_datetime(last_dt, errors="coerce")
        if pd.isna(t):
            t = pd.Timestamp("2000-01-01")
        row: dict[str, Any] = {
            "CustomerID": int(customer_id),
            "ProductID": pid,
            "OrderDate": t,
            "ProductCategoryName": str(pr.get("ProductCategoryName", "unknown")),
            "ProductSubCategoryName": str(pr.get("ProductSubCategoryName", "unknown")),
            "ProductModelName": str(pr.get("ProductModelName", "unknown")),
            "Color": str(pr.get("Color", "unknown")),
            "ListPrice": float(pr.get("ListPrice", 0.0) or 0.0),
            "City": str(cr.get("City", "unknown")),
            "StateProvinceName": str(cr.get("StateProvinceName", "unknown")),
            "CountryName": str(cr.get("CountryName", "unknown")),
            "order_year": int(t.year),
            "order_month": int(t.month),
            "order_dow": int(t.dayofweek),
            "user_train_lines_total": nlines,
            "product_train_line_count": float(p_counts.get(pid, 0.0)),
            "user_order_seq": uo,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_clf_extras(
    path: str,
    *,
    product_dim: pd.DataFrame,
    customer_dim: pd.DataFrame,
    user_stats: dict[int, Any],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "product_dim": product_dim,
                "customer_dim": customer_dim,
                "user_stats": user_stats,
            },
            f,
        )


def load_clf_extras(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_user_stats_from_train(train_df: pd.DataFrame) -> dict[int, Any]:
    """Per-customer: n train lines, last order date, product_train_count dict, seq proxy."""
    if train_df.empty:
        return {}
    t = train_df.sort_values(
        ["CustomerID", "OrderDate", "SalesOrderID"],
        kind="mergesort",
    )
    nlines = t.groupby("CustomerID").size()
    last = t.groupby("CustomerID")["OrderDate"].max()
    out: dict[int, Any] = {}
    for u in t["CustomerID"].unique():
        u = int(u)
        subp = t[t["CustomerID"] == u]["ProductID"].value_counts()
        out[u] = {
            "n_train_lines": float(nlines.get(u, 0)),
            "last_order": last.get(u, pd.NaT),
            "user_order_seq": float(nlines.get(u, 0)) + 1.0,
            "product_train_count": {int(k): int(v) for k, v in subp.items()},
        }
    return out


def product_customer_dims_for_inference(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per product / customer for inference (shared by training scripts)."""
    pcols = [
        "ProductID",
        "ProductCategoryName",
        "ProductSubCategoryName",
        "ProductModelName",
        "Color",
        "ListPrice",
    ]
    product_dim = (
        base[pcols].drop_duplicates("ProductID")
        if all(c in base.columns for c in pcols)
        else base[["ProductID"]].drop_duplicates()
    )
    ccols = ["CustomerID", "City", "StateProvinceName", "CountryName"]
    customer_dim = (
        base[ccols].drop_duplicates("CustomerID")
        if all(c in base.columns for c in ccols[1:])
        else base[["CustomerID"]]
        .drop_duplicates()
        .assign(
            City="unknown",
            StateProvinceName="unknown",
            CountryName="unknown",
        )
    )
    return product_dim, customer_dim


def write_rec_and_clf_artifacts(
    data_dir: str,
    base: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    """
    Minimal interaction lines → `prepare_interaction_split` → `interaction_meta.pkl`,
    `clf_extras.pkl`, and `training_stats.json`.
    """
    from data_prep import (
        prepare_interaction_split,
        write_interaction_metadata_pickle,
        write_train_interaction_stats_json,
    )

    cols = [c for c in ["CustomerID", "ProductID", "OrderDate", "SalesOrderID", "LineTotal"] if c in base.columns]
    imin = base[cols].copy()
    if "SalesOrderID" not in imin.columns:
        imin["SalesOrderID"] = range(len(imin))
    if "LineTotal" not in imin.columns:
        imin["LineTotal"] = 1.0
    tr_df, _, meta, train_csr = prepare_interaction_split(imin)
    ustats = build_user_stats_from_train(tr_df)
    product_dim, customer_dim = product_customer_dims_for_inference(base)
    save_clf_extras(
        os.path.join(data_dir, "clf_extras.pkl"),
        product_dim=product_dim,
        customer_dim=customer_dim,
        user_stats=ustats,
    )
    write_interaction_metadata_pickle(
        os.path.join(data_dir, "interaction_meta.pkl"),
        meta,
    )
    write_train_interaction_stats_json(
        tr_df,
        os.path.join(data_dir, "training_stats.json"),
    )
    return meta, tr_df, train_csr
