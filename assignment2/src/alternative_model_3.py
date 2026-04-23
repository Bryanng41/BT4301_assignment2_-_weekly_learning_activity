"""Alternative 3: LightGBM on classification event table."""
from __future__ import annotations

import os
import sys

import mlflow

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))

from classification_data import (
    load_enriched_interactions,
    load_or_build_event_tables,
    save_clf_feature_config,
    save_product_customer_dims,
)
from clf_modeling import (
    fit_clf,
    get_xy,
    predict_metrics,
    write_rec_and_clf_artifacts,
)

if __name__ == "__main__":
    _here = os.path.dirname(__file__)
    _data = os.path.join(_here, "..", "data")
    os.makedirs(_data, exist_ok=True)

    tr_ev, te_ev = load_or_build_event_tables(_data, n_neg=5, random_state=42)
    save_clf_feature_config(os.path.join(_data, "clf_feature_config.json"))
    base = load_enriched_interactions()
    save_product_customer_dims(base, _data)
    write_rec_and_clf_artifacts(_data, base)

    pipeline, th, _ = fit_clf("lgbm", tr_ev)
    m = predict_metrics(pipeline, te_ev, threshold=th)
    acc, prec, rec, f1 = m["accuracy"], m["precision"], m["recall"], m["f1"]
    print("LightGBM test metrics:", acc, prec, rec, f1)

    from mlflow_util import (
        log_sklearn_pipeline_model,
        set_mlflow_tracking,
        sklearn_param_dict_for_logging,
    )

    set_mlflow_tracking()
    X_train, _y_tr = get_xy(tr_ev.copy())
    mlflow.set_experiment("Assignment2 - Alternative Model 3")
    with mlflow.start_run():
        mlflow.log_params(
            sklearn_param_dict_for_logging(
                pipeline,
                method="lightgbm",
                n_negatives=5,
                proba_threshold=th,
            )
        )
        log_sklearn_pipeline_model(
            pipeline,
            X_train,
            artifact_path="purchase_clf",
            registered_model_name="assignment2_alt3_lgbm",
        )
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.set_tag("Training Info", "LightGBM + OHE/scale (purchase yes/no)")

    print("MLflow run complete.")
