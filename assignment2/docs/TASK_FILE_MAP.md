# Assignment 2 — Files by task

This document maps project files to **Task A** (problem formulation), **Task B** (model development + MLflow), **Task C** (Docker + REST API), and **Task D** (web app + monitoring). Paths are relative to the `assignment2/` project root.

---

## Task A — Business problem & formulation

| File | Role |
|------|------|
| `docs/TASK_A_problem_formulation.md` | Problem statement, analytics approach, how MLflow/export/Docker/web app tie to the business question. |
| `guide.md` | End-to-end runbook (data, training, MLflow, export, Docker, Streamlit, ports). |
| `docs/IMPLEMENTATION_MAP.md` | Maps the source tree to Tasks A–D and submission notes. |

---

## Task B — Model development & MLflow

### Data & feature pipeline

| File | Role |
|------|------|
| `src/data_prep.py` | Loads interaction lines (MySQL or `data/raw_data.csv`), **temporal train/test split**, builds **user×item** metadata (`recsys_meta.pkl`), `raw_data.csv`, `training_stats.json`, etc. |
| `src/classification_data.py` | Joins warehouse features, builds **enriched** rows, **`train_events.csv` / `test_events.csv`** with **sampled negatives**, saves `clf_extras.pkl` / feature config when scripts run. Defines **`CATEGORICAL_FEATURES`** and **`NUMERIC_FEATURES`**. |

### Shared modeling & logging

| File | Role |
|------|------|
| `src/clf_modeling.py` | Sklearn **pipelines** (logistic regression, random forest, XGBoost, LightGBM), **tuned probability threshold** (non-LR), **`get_xy` / `predict_metrics`**, **`fit_clf`**, **`write_rec_and_clf_artifacts`**, inference row builder for serving. |
| `src/mlflow_util.py` | Default **MLflow tracking URI** (`MLFLOW_TRACKING_URI` or `http://127.0.0.1:9080`), **`log_sklearn_pipeline_model`**, **`sklearn_param_dict_for_logging`**. |

### Experiments (one script per MLflow experiment)

| File | Role |
|------|------|
| `src/baseline_model.py` | **Baseline:** logistic regression on `train_events`, metrics on `test_events`, logs to **Assignment2 - Baseline Model**. |
| `src/alternative_model_1.py` | **Alternative 1:** random forest — same data/metrics path, **Assignment2 - Alternative Model 1**. |
| `src/alternative_model_2.py` | **Alternative 2:** XGBoost — **Assignment2 - Alternative Model 2**. |
| `src/alternative_model_3.py` | **Alternative 3:** LightGBM — **Assignment2 - Alternative Model 3**. |

### Model selection & production bundle (feeds Task C)

| File | Role |
|------|------|
| `src/evaluate_models.py` | Reads **latest run per experiment** in MLflow, picks **highest F1**, **refits** the winner on full `train_events`, writes **`docker/model/serve_bundle.pkl`**, `recsys_meta.pkl`, `model_card.json`, copies stats/CSVs to `webapp/`. |

### Task B notebooks (evidence / analysis)

| File | Role |
|------|------|
| `notebooks/threshold_logistic_regression.ipynb` | Exploratory work on LR threshold (and related); **include saved outputs** if the brief asks for Task B figures in a notebook. |
| `notebooks/eda_customer_purchase_counts.ipynb` | EDA example; include if part of your Task B narrative. |
| `notebooks/README_kernel.md` | Notes for notebook kernel/env (optional). |

### Supporting library for inference (used by Docker after export)

| File | Role |
|------|------|
| `src/purchase_prediction.py` | **`predict_purchase_for_pair`** (tabular P(purchase) + threshold) and **`EXPERIMENT_KIND`** mapping MLflow experiment names → serve `kind`. Copied next to the API in the container. |

---

## Task C — Docker & REST API

| File | Role |
|------|------|
| `docker/Dockerfile` | Builds the API image: Python slim, installs `docker/requirements.txt`, copies app + `purchase_prediction`, `clf_modeling`, `classification_data`, `docker/model/`. |
| `docker/requirements.txt` | **Pip dependencies** for the API container (Connexion, pandas, scikit-learn, xgboost, lightgbm, etc.). |
| `docker/compose.yaml` | **`docker compose`** service (e.g. host port **5002** → container **5000**). |
| `docker/.dockerignore` | Reduces Docker build context size. |
| `docker/smoke_api.sh` | Helper script to curl health/purchase when `localhost` is wrong (e.g. dev container). |
| `docker/src/server.py` | **Connexion + Flask** entry: loads OpenAPI spec, serves routes. |
| `docker/src/predict.yml` | **OpenAPI 2** spec: `GET /api/purchase`, response schema. |
| `docker/src/predict.py` | Loads **`serve_bundle.pkl`**, calls **`predict_purchase_for_pair`**, returns JSON (or 400 for bad input). |
| `docker/src/encode.py` | Placeholder / small helpers (optional). |

### Artifacts produced for the container (after `evaluate_models.py`)

| File | Role |
|------|------|
| `docker/model/serve_bundle.pkl` | **Exported winner:** trained pipeline, `kind`, `meta`, `clf_extras`, `clf_threshold`. |
| `docker/model/recsys_meta.pkl` | Copy of recommender/event **metadata** (id maps, catalog size, etc.). |
| `docker/model/model_card.json` | **Traceability:** best experiment, run id, F1, comparison rows. |

---

## Task D — Web application & monitoring

| File | Role |
|------|------|
| `webapp/app.py` | **Streamlit** app: **Purchase prediction** tab → `GET /api/purchase` on configurable base URL; **Drift** tab → KS tests vs `raw_data.csv`, ground-truth-style line from history, optional `training_stats.json`. |
| `webapp/requirements.txt` | **Pip dependencies** for Streamlit, pandas, scipy, requests, etc. |
| `webapp/simulate_drift.py` | Optional script to seed or demo drift scenarios (if you use it in the report). |

### Webapp data (optional / generated)

| File | Role |
|------|------|
| `webapp/raw_data.csv` | **Drift reference** (copied from training pipeline by `evaluate_models.py`). |
| `webapp/training_stats.json` | Optional JSON for the dashboard expander. |
| `webapp/input_monitoring_log.csv` | **Append-only log** of API monitoring fields per prediction (created when users run predictions). |
| `webapp/production_log.csv` | Legacy log path; app may fall back if present. |

---

## Dependency recreation (submit; do not submit `.venv`)

| File | Role |
|------|------|
| `docker/requirements.txt` | Recreate **API** environment. |
| `webapp/requirements.txt` | Recreate **Streamlit** environment. |
| `guide.md` | **Training** install one-liners (`pip install …` for MySQL, MLflow, xgboost, lightgbm, scikit-learn, etc.). |

---

## Data & large folders (policy-dependent)

| Path | Role |
|------|------|
| `data/` | `train_events.csv`, `test_events.csv`, `raw_data.csv`, `clf_extras.pkl`, etc. — **regenerated** by `data_prep.py` / training scripts if markers run the pipeline. Include only if your course requires prebuilt CSVs. |
| `mlruns/`, `mlartifacts/` | MLflow local store — often **omitted** from submission; markers can re-run training or use `file:` URI per `guide.md`. |

---

## Quick task → folder summary

| Task | Primary locations |
|------|-------------------|
| **A** | `docs/TASK_A_problem_formulation.md`, `guide.md`, `docs/IMPLEMENTATION_MAP.md` |
| **B** | `src/*.py` (pipeline + models + `evaluate_models.py`), `notebooks/*.ipynb` |
| **C** | `docker/` (including `docker/model/` after export) |
| **D** | `webapp/` |
