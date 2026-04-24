# Implementation map (Assignment 2 deliverables)

This document ties your **source tree** to **Tasks A–D** and the submission checklist: Python/ notebooks, Docker (Task C), web app (Task D), and dependency manifests. It does **not** list third-party code (e.g. `.venv`, `node_modules`).

**Operational runbook (commands, Docker, Postman, ports):** see `../guide.md`.

---

## What to submit vs omit

| Include | Omit (typical) |
|--------|-----------------|
| `.py`, `.ipynb` you authored; `docker/`; `webapp/`; `docs/`; `src/`; `notebooks/`; root `guide.md` if your course wants it | `.venv/`, `__pycache__/`, `.git` |
| `docker/requirements.txt`, `webapp/requirements.txt`; training deps are often listed in `guide.md` or a root `requirements.txt` if you add one | Huge MLflow artifact trees under `src/mlartifacts/` unless the brief requires them—many teams submit **exported** `docker/model/` only |
| `docker/model/serve_bundle.pkl` and related exports needed to **build/run** the API | Regenerated CSVs under `data/` if policy says “repro from scripts only” |

Confirm exact zip layout with your instructor.

---

## Task A — Business problem & formulation

| File | Role |
|------|------|
| `docs/TASK_A_problem_formulation.md` | Problem/challenge/opportunity, how recommendations address it, and how success might be judged. |
| `src/data_prep.py` | Pulls interaction-level data from MySQL, builds **temporal 80/20 per-customer** split, writes train/test interactions, **`raw_data.csv`**, **`interaction_meta.pkl`**, **`training_stats.json`**, etc. Feeds all modeling and drift reference. If product names cannot be loaded from the warehouse, **id-based** display names are used when building metadata. |
| `src/classification_data.py` | **Enriched** line-level features (product + customer + time), **train/test 0-1** event tables (**5 negatives per user** in train, same in test, sampled from the train product pool), optional **`data/train_events.csv`**, **`data/test_events.csv`**, **`data/clf_extras.pkl`**, **`clf_feature_config.json`**. |
| `src/clf_modeling.py` | Sklearn **pipelines** (OHE + scaling + LR / RF / XGB / LightGBM), **tuned proba threshold** for reporting, **inference** row builder for the API. |
| `notebooks/investigate_orderqty_target.ipynb` | Exploratory / investigative work (often cited in Task A or B); may include plots or narrative. |
| `data/assignment_1_transform.ipynb` | Optional/legacy notebook if still part of your story (Assignment 1 warehouse path); include only if it is part of your deliverable narrative. |

---

## Task B — Deeper analysis / modeling setup (course-dependent)

| File | Role |
|------|------|
| `notebooks/investigate_orderqty_target.ipynb` | Primary place for **sample outputs** (text, tables, figures) if the brief asks for Task B evidence inside a notebook. |
| `src/data_prep.py` | Defines the **interaction matrix** and evaluation pairs used downstream. |
| `src/purchase_prediction.py` | **`predict_purchase_for_pair`** (tabular **P(purchase)** + threshold) and **`EXPERIMENT_KIND`** (MLflow experiment name → serve `kind`). |

---

## Task C — Best model in Docker + REST API

| File | Role |
|------|------|
| `docker/Dockerfile` | Builds the API image from **python:3.12-slim**, pip installs from `docker/requirements.txt`, copies `docker/src/`, `purchase_prediction.py`, **`clf_modeling.py`**, **`classification_data.py`**, and `docker/model/`. |
| `docker/compose.yaml` | Runs the `api` service; maps host **`5002` → container `5000`**. |
| `docker/requirements.txt` | Pip deps for the **container** (Connexion, pandas, scikit-learn, xgboost, lightgbm, etc.). |
| `docker/.dockerignore` | Shrinks build context / speeds builds. |
| `docker/src/server.py` | Connexion **Flask** entrypoint: loads `predict.yml`, **`/api/health`**, runs `app.run`. |
| `docker/src/predict.yml` | OpenAPI 2 spec: **`GET /api/purchase`** (`customer_id`, `product_id`) and schemas; Swagger UI path is Connexion’s default (e.g. `/ui/`). |
| `docker/src/predict.py` | Loads **`serve_bundle.pkl`**, calls **`predict_purchase_for_pair`** from `purchase_prediction`, returns JSON (incl. **cold_start_user**). |
| `docker/model/serve_bundle.pkl` | **Exported winner** (tabular model + meta + extras) for inference—produced by `evaluate_models.py`. |
| `docker/model/interaction_meta.pkl`, `docker/model/model_card.json` | Metadata / card copied with the bundle for traceability. |

**Training / export (feeds Task C artifact):**

| File | Role |
|------|------|
| `src/baseline_model.py` | **Logistic regression** on **`train_events`**, test metrics: **accuracy / precision / recall / f1** (MLflow; primary **f1** for selection). |
| `src/alternative_model_1.py` | **Random forest** + same metrics. |
| `src/alternative_model_2.py` | **XGBoost** + same metrics. |
| `src/alternative_model_3.py` | **LightGBM** on **`train_events`**, test metrics on **`test_events`**, MLflow. |
| `src/evaluate_models.py` | Compares runs (**highest f1** on **`test_events`**), **refits** the winning **pipeline**, writes **`serve_bundle.pkl`** (**`clf_extras`**, **`clf_threshold`**, **meta**, **`kind`**) and copies reference CSVs to **`webapp/`**. |

---

## Task D — Web app + monitoring / feedback loop

| File | Role |
|------|------|
| `webapp/app.py` | **Streamlit** app: **Purchase prediction** tab calls **`GET /api/purchase`** on configurable base URL; **Drift dashboard** compares **`raw_data.csv`** (training reference) vs **`production_log.csv`** (post-production inputs) with **KS tests**, traffic light, optional **`training_stats.json`** expander. |
| `webapp/requirements.txt` | Pip deps for Streamlit, pandas, scipy, requests, etc. |
| `webapp/raw_data.csv` | Drift **reference** (copied by `evaluate_models.py` from training pipeline). |
| `webapp/training_stats.json` | Optional JSON shown in dashboard expander. |
| `webapp/production_log.csv` | **Append-only production log** when users request recommendations (or seeded rows for demos); drives drift tab. |

---

## Dependency artifacts (recreate env without `.venv`)

| File | Role |
|------|------|
| `docker/requirements.txt` | API container. |
| `webapp/requirements.txt` | Streamlit app. |
| `guide.md` (project root) | Documents full training stack install (MySQL connector, MLflow, lightgbm, xgboost, etc.); use or replace with a single **`requirements-training.txt`** if you want one file for markers. |

---

## Suggested reading order (to understand the full implementation)

1. `docs/TASK_A_problem_formulation.md` — business framing.  
2. `src/data_prep.py` — data you model on.  
3. `src/purchase_prediction.py` — pair-level **P(purchase)** serving helper.  
4. `src/baseline_model.py` … `alternative_model_3.py` — experiments.  
5. `src/evaluate_models.py` — winner selection and **Docker/webapp** artifacts.  
6. `docker/src/server.py`, `predict.yml`, `predict.py` — REST contract and inference.  
7. `webapp/app.py` — Task D UI and drift dashboard.  
8. `guide.md` — how to run everything end-to-end.

---

## Notebooks and Task B outputs

If the brief requires **Task B outputs inside the notebook**, open **`notebooks/investigate_orderqty_target.ipynb`** and ensure key cells are **executed** so markers see tables/figures without re-running. Keep outputs course-appropriate (size and privacy).
