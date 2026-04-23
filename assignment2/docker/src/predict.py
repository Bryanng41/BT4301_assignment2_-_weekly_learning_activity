import os
import sys

import joblib

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)

from purchase_prediction import predict_purchase_for_pair

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(os.path.join(_BASE, "..", "model", "serve_bundle.pkl"))


def purchase(customer_id, product_id):
    """
    Query params from OpenAPI: predict purchase (yes/no) for one customer–product pair.
    """
    _load()
    cid = int(customer_id)
    pid = int(product_id)
    try:
        out = predict_purchase_for_pair(
            _bundle["kind"],
            _bundle["model"],
            cid,
            pid,
            _bundle["meta"],
            clf_extras=_bundle.get("clf_extras"),
            clf_threshold=_bundle.get("clf_threshold"),
        )
    except ValueError as e:
        return {"detail": str(e)}, 400
    return out
