"""
Compare Assignment 2 MLflow runs (tabular classifiers) and export the best model.

Selects the experiment with the highest logged f1 on the test event table.
"""
from __future__ import annotations

import json
import os
import shutil

import joblib
import pandas as pd
from mlflow.tracking import MlflowClient

from clf_modeling import fit_clf, load_clf_extras
from data_prep import (
    load_interactions,
    prepare_interaction_split,
    write_interaction_metadata_pickle,
)
from purchase_prediction import EXPERIMENT_KIND

from mlflow_util import set_mlflow_tracking

set_mlflow_tracking()
client = MlflowClient()

METRIC_KEY = "f1"

EXPERIMENTS = [
    ("Assignment2 - Baseline Model", "baseline_model", "Logistic regression (tabular)"),
    ("Assignment2 - Alternative Model 1", "alternative_model_1", "Random forest (tabular)"),
    ("Assignment2 - Alternative Model 2", "alternative_model_2", "XGBoost (tabular)"),
    ("Assignment2 - Alternative Model 3", "alternative_model_3", "LightGBM (tabular)"),
]


def main():
    best = None
    rows: list[dict] = []
    for exp_name, artifact_subdir, desc in EXPERIMENTS:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            print(f"Missing experiment: {exp_name}")
            continue
        runs = client.search_runs(
            [exp.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=1,
        )
        if not runs:
            print(f"No runs in {exp_name}")
            continue
        r = runs[0]
        m = r.data.metrics
        f1 = float(m.get("f1", 0.0))
        acc = float(m.get("accuracy", 0.0))
        prec = float(m.get("precision", 0.0))
        rec = float(m.get("recall", 0.0))
        rows.append(
            {
                "experiment": exp_name,
                "run_id": r.info.run_id,
                "f1": f1,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "description": desc,
                "artifact_subdir": artifact_subdir,
            }
        )
        if best is None or f1 > best["f1"]:
            best = {**rows[-1]}

    print(f"\n=== Model comparison (latest run per experiment; higher {METRIC_KEY} is better) ===")
    for row in sorted(rows, key=lambda x: x["f1"], reverse=True):
        print(
            f"  {row['experiment']}: f1={row['f1']:.4f} "
            f"acc={row['accuracy']:.4f} prec={row['precision']:.4f} rec={row['recall']:.4f}"
        )

    if not best:
        raise SystemExit("No runs found.")

    print(f"\n** Best for production (highest f1): {best['experiment']} **")
    print(f"   MLflow run ID: {best['run_id']}")

    _root = os.path.join(os.path.dirname(__file__), "..")
    _data = os.path.join(_root, "data")
    train_path = os.path.join(_data, "train_events.csv")
    extras_path = os.path.join(_data, "clf_extras.pkl")
    if not os.path.isfile(train_path):
        raise SystemExit(
            f"Missing {train_path}"
        )
    tr_ev = pd.read_csv(train_path)
    imin = load_interactions()
    if "SalesOrderID" not in imin.columns:
        imin["SalesOrderID"] = range(len(imin))
    if "LineTotal" not in imin.columns:
        imin["LineTotal"] = 1.0
    imin2 = imin[[c for c in ["CustomerID", "ProductID", "OrderDate", "SalesOrderID", "LineTotal"] if c in imin.columns]].copy()
    _train_df, _test_df, meta, _train_csr = prepare_interaction_split(imin2)
    kind = EXPERIMENT_KIND[best["experiment"]]

    if kind not in ("log_reg", "rf", "xgb", "lgbm"):
        raise ValueError(f"Unknown production kind: {kind}")

    model, clf_threshold, _ = fit_clf(kind, tr_ev)
    clf_extras = None
    if os.path.isfile(extras_path):
        clf_extras = load_clf_extras(extras_path)

    out_dir = os.path.join(_root, "docker", "model")
    os.makedirs(out_dir, exist_ok=True)

    bundle = {
        "kind": kind,
        "model": model,
        "meta": meta,
        "best_experiment": best["experiment"],
        "clf_extras": clf_extras,
        "clf_threshold": clf_threshold,
    }
    joblib.dump(bundle, os.path.join(out_dir, "serve_bundle.pkl"))

    write_interaction_metadata_pickle(os.path.join(out_dir, "interaction_meta.pkl"), meta)

    card = {
        "task": "purchase_classification",
        "primary_metric": METRIC_KEY,
        "best_experiment": best["experiment"],
        "best_run_id": best["run_id"],
        "best_f1_mlflow": best["f1"],
        "recommender_kind": kind,
        "comparison": rows,
    }
    with open(os.path.join(out_dir, "model_card.json"), "w") as f:
        json.dump(card, f, indent=2)

    stats_src = os.path.join(_data, "training_stats.json")
    stats_dst = os.path.join(_root, "webapp", "training_stats.json")
    os.makedirs(os.path.dirname(stats_dst), exist_ok=True)
    if os.path.isfile(stats_src):
        shutil.copy2(stats_src, stats_dst)

    raw_src = os.path.join(_data, "raw_data.csv")
    raw_dst = os.path.join(_root, "webapp", "raw_data.csv")
    if os.path.isfile(raw_src):
        shutil.copy2(raw_src, raw_dst)

    print(f"\nExported: {out_dir}/serve_bundle.pkl, interaction_meta.pkl, model_card.json")


if __name__ == "__main__":
    main()
