"""
Pair-level purchase prediction helpers for the Docker API and model export.

Served models are sklearn ``Pipeline``s (logistic regression, random forest, XGBoost, LightGBM)
with ``predict_proba`` and a tuned decision threshold.
"""
from __future__ import annotations

from typing import Any

from clf_modeling import ClfKind, build_inference_rows_for_user, get_X_only

TABULAR_KINDS = frozenset({"log_reg", "rf", "xgb", "lgbm"})

EXPERIMENT_KIND: dict[str, ClfKind] = {
    "Assignment2 - Baseline Model": "log_reg",
    "Assignment2 - Alternative Model 1": "rf",
    "Assignment2 - Alternative Model 2": "xgb",
    "Assignment2 - Alternative Model 3": "lgbm",
}


def predict_purchase_for_pair(
    kind: str,
    model_obj: Any,
    customer_id: int,
    product_id: int,
    meta: dict[str, Any],
    *,
    clf_extras: dict[str, Any] | None = None,
    clf_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Return purchase prediction for one (customer_id, product_id) pair: P(purchase) and threshold.
    """
    pii = meta["product_id_to_idx"]
    pid = int(product_id)
    if pid not in pii:
        raise ValueError(f"unknown product_id {pid}")
    j = int(pii[pid])
    name = str(meta.get("product_id_to_name", {}).get(str(pid), str(pid)))
    base = {
        "customer_id": int(customer_id),
        "product_id": pid,
        "product_name": name,
        "kind": kind,
    }

    if kind not in TABULAR_KINDS:
        raise ValueError(f"unsupported kind {kind!r}")
    if clf_extras is None:
        raise ValueError("missing clf_extras")

    rows = build_inference_rows_for_user(
        int(customer_id),
        meta,
        clf_extras["product_dim"],
        clf_extras["customer_dim"],
        clf_extras["user_stats"],
    )
    row_j = rows.iloc[j : j + 1]
    Xp = get_X_only(row_j)
    proba = float(model_obj.predict_proba(Xp)[0, 1])
    th = float(clf_threshold) if clf_threshold is not None else 0.0
    cold = int(customer_id) not in meta["customer_id_to_idx"]
    return {
        **base,
        "score": proba,
        "threshold": th,
        "predicted_purchase": proba >= th,
        "cold_start_user": cold,
    }
