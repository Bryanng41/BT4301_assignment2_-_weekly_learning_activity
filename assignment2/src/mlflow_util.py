"""MLflow: tracking URI and helpers to log sklearn pipelines (course style).

Precedence: ``MLFLOW_TRACKING_URI`` if set, else the class tracking server on
port 9080. For offline / file-only use: set
``export MLFLOW_TRACKING_URI=file:/absolute/path/to/assignment2/mlruns``
before training.
"""
from __future__ import annotations

import os
from typing import Any

import mlflow
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

# Default class MLflow server (see guide)
_DEFAULT_TRACKING_URI = "http://127.0.0.1:9080"

try:
    from mlflow.exceptions import MlflowException
except ImportError:  # pragma: no cover
    MlflowException = Exception  # type: ignore[misc, assignment]


def set_mlflow_tracking() -> None:
    """Point MLflow at ``MLFLOW_TRACKING_URI`` if set, else ``http://127.0.0.1:9080``."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
        return
    mlflow.set_tracking_uri(_DEFAULT_TRACKING_URI)


def _param_val(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, (bool, int, float, str)):
        return str(v)
    return repr(v)


def sklearn_param_dict_for_logging(
    pipeline: Pipeline,
    *,
    method: str,
    n_negatives: int,
    proba_threshold: float,
) -> dict[str, str]:
    """Small flat dict of string values for ``mlflow.log_params`` (course tutorial style)."""
    out: dict[str, str] = {
        "method": method,
        "n_negatives_per_user_train": str(n_negatives),
        "proba_threshold": f"{proba_threshold:.6f}",
    }
    clf = pipeline.named_steps.get("clf")
    if clf is not None:
        name = type(clf).__name__
        out["estimator"] = name
        if name == "LogisticRegression":
            out["clf_max_iter"] = _param_val(getattr(clf, "max_iter", None))
            out["clf_C"] = _param_val(getattr(clf, "C", None))
        elif name == "RandomForestClassifier":
            out["clf_n_estimators"] = _param_val(getattr(clf, "n_estimators", None))
            out["clf_max_depth"] = _param_val(getattr(clf, "max_depth", None))
        elif name == "XGBClassifier":
            out["clf_n_estimators"] = _param_val(getattr(clf, "n_estimators", None))
            out["clf_max_depth"] = _param_val(getattr(clf, "max_depth", None))
            out["clf_learning_rate"] = _param_val(getattr(clf, "learning_rate", None))
        elif name == "LGBMClassifier":
            out["clf_n_estimators"] = _param_val(getattr(clf, "n_estimators", None))
            out["clf_max_depth"] = _param_val(getattr(clf, "max_depth", None))
            out["clf_learning_rate"] = _param_val(getattr(clf, "learning_rate", None))
    return out


def log_sklearn_pipeline_model(
    pipeline: Pipeline,
    X_train: Any,
    *,
    artifact_path: str,
    registered_model_name: str,
) -> None:
    """
    Log fitted sklearn ``Pipeline`` with signature and input example.
    If Model Registry registration fails, log the model without ``registered_model_name``.
    """
    ex = X_train.head(5) if hasattr(X_train, "head") else X_train
    y_hat = pipeline.predict(ex)
    signature = infer_signature(ex, y_hat)
    try:
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=artifact_path,
            signature=signature,
            input_example=ex,
            registered_model_name=registered_model_name,
        )
    except (MlflowException, OSError) as e:
        err = str(e).lower()
        if any(
            s in err
            for s in (
                "registered",
                "registry",
                "forbidden",
                "unauthorized",
                "409",
                "400",
            )
        ):
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=artifact_path,
                signature=signature,
                input_example=ex,
            )
        else:
            try:
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=ex,
                )
            except (MlflowException, OSError):
                mlflow.set_tag(
                    "mlflow_model_skipped",
                    "log_model failed; metrics still logged. Check MLFLOW_TRACKING_URI / server.",
                )
