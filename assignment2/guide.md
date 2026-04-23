# Assignment 2 Guide

## Overview

This folder contains the Assignment 1 **Adventure Works** star-schema warehouse, a **supervised (binary) purchase** **classification** pipeline on **(customer, product)** event rows, MLflow experiments, Docker-hosted **Connexion** REST inference for **per (customer, product) purchase** prediction, and a **Streamlit** web app for calling the API and monitoring input drift. All four alternative models are **tabular** classifiers served as **P(purchase)** with a tuned threshold.

End-to-end workflow:

1. Prepare interaction data from MySQL (temporal **80/20 per-customer** split, user×item matrix metadata).
2. Train four models (**logistic regression**, **random forest**, **XGBoost**, **LightGBM**) on **`train_events`**; log **metrics**, **params**, and **sklearn Pipeline** models to MLflow (default tracking: **`http://127.0.0.1:9080`** unless **`MLFLOW_TRACKING_URI`** is set; use a **`file:…/mlruns`** URI for a local file store when no server is running).
3. Compare runs (best **f1**), refit the winner, export `docker/model/*` and copy artifacts into **`webapp/`**.
4. Run the **purchase prediction API** in Docker (build context is **`assignment2/`** root; container serves on **5000**, host maps **5002**).
5. Run the **Streamlit** web app; point it at the API base URL.

## Folder structure

| Path | Role |
|------|------|
| `src/data_prep.py` | MySQL (or `raw_data.csv` fallback) interaction extract, temporal split, `recsys_meta.pkl`, `interactions_*.csv`, `raw_data.csv`, `training_stats.json`. |
| `src/classification_data.py` | Enriched features and **`train_events.csv` / `test_events.csv`**, five negatives per user, `clf_extras` / `dim` pickles as built by training scripts. |
| `src/clf_modeling.py` | Sklearn **pipelines** (LR, RF, XGB, LightGBM) and **inference** row builder for tabular models. |
| `src/mlflow_util.py` | Default MLflow **`http://127.0.0.1:9080`** if **`MLFLOW_TRACKING_URI`** is unset; helpers to **`mlflow.log_model`** (tabular sklearn **Pipeline**). Set **`file:…/assignment2/mlruns`** for file-only (offline) tracking. |
| `src/purchase_prediction.py` | **`predict_purchase_for_pair`** and **`EXPERIMENT_KIND`** (MLflow experiment name → `kind` string for export). |
| `src/baseline_model.py` | **Logistic regression** on events + MLflow. |
| `src/alternative_model_*.py` | **Random forest** / **XGBoost** / **LightGBM** + MLflow. |
| `src/evaluate_models.py` | Picks best by **highest f1** (test event table), refits, exports `docker/model/serve_bundle.pkl`, `recsys_meta.pkl`, `model_card.json`; copies **`training_stats.json`** and **`raw_data.csv`** into **`webapp/`**. |
| `data/` | See above, plus `train_events.csv`, `test_events.csv`, `clf_*` as generated. |
| `docker/` | `Dockerfile` (context `..`), `compose.yaml`, `requirements.txt`, **`src/`** (app + copies of `purchase_prediction.py`, `clf_modeling.py`, `classification_data.py`), **`model/`** (bundle from `evaluate_models.py`). |
| `webapp/` | Streamlit `app.py`; receives copies of `raw_data.csv` / `training_stats.json`; writes `production_log.csv`. |
| `docs/` | Task A problem formulation (`TASK_A_problem_formulation.md`). |

## Prerequisites

- **Python 3.12** and a project venv. Install training deps (includes **`lightgbm`**, **`xgboost`**, **`scipy`**):

  ```bash
  cd assignment2
  pip install mysql-connector-python pandas numpy scipy scikit-learn xgboost lightgbm mlflow joblib
  ```

- **MySQL** with Assignment 1 warehouse: host `localhost`, database **`datawarehouse`**, user **`root`** (adjust `DB_CONFIG` in `src/data_prep.py` if yours differs). If MySQL is down, **`data/raw_data.csv`** is used to build splits and event tables.
- **MLflow:** Training defaults to the tracking server at **`http://127.0.0.1:9080`**. For a **local file store** only (no server), set e.g. **`export MLFLOW_TRACKING_URI=file:/absolute/path/to/assignment2/mlruns`** (see `src/mlflow_util.py`). Start **`mlflow server --port 9080`** *before* training if you use the default URI.
- **Docker** (optional) to run the API container.
- **Streamlit**: **`webapp/requirements.txt`**.

---

## How to test everything (end-to-end)

Run steps **in order**. Each checkpoint tells you what “success” looks like.

### 0. Services and venv

1. **MySQL** — the warehouse must accept connections with the same settings as `DB_CONFIG` in `src/data_prep.py` (default: `localhost`, user `root`, database `datawarehouse`). Quick check:

   ```bash
   mysql -u root -h localhost datawarehouse -e "SELECT COUNT(*) FROM sales;"
   ```

2. **MLflow** — by default, scripts log to **http://127.0.0.1:9080**. In a **separate terminal** start the server, then run training (or set **`MLFLOW_TRACKING_URI=file:…/mlruns`** to log locally without a server):

   ```bash
   mlflow server --host 0.0.0.0 --port 9080
   ```

   Open **http://localhost:9080** and confirm the UI loads. After training, you should see four experiments (Baseline + three alternatives) with **metrics** and **Models** (logged sklearn pipelines). If the server is not running and you did **not** set a **`file:`** tracking URI, training may error when writing to MLflow; use **`file:`** for offline work.

3. **Python venv** — from `assignment2/`, install deps (includes `lightgbm` and `xgboost`):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install mysql-connector-python pandas numpy scipy scikit-learn xgboost lightgbm mlflow joblib
   ```

### 1. Data prep

```bash
cd /path/to/assignment2/src
python data_prep.py
```

**Check:** `../data/interactions_train.csv`, `interactions_test.csv`, `raw_data.csv`, and `recsys_meta.pkl` exist. The script prints train/test line counts and the number of evaluation **test pairs**.

### 2. Train all four models

Still in `src/`:

```bash
python baseline_model.py
python alternative_model_1.py
python alternative_model_2.py
python alternative_model_3.py
```

**Check:** Each run prints **accuracy, precision, recall, f1** (and logs **`proba_threshold`**) to the console. In MLflow, open each experiment’s latest run and confirm those metrics and parameters.

### 3. Export the best model for Docker / Streamlit

```bash
python evaluate_models.py
```

**Check:**

- Console prints a comparison table (**higher f1 is better**) and names the winning experiment.
- Files exist: `docker/model/serve_bundle.pkl`, `docker/model/recsys_meta.pkl`, `docker/model/model_card.json`.
- `webapp/raw_data.csv` and `webapp/training_stats.json` were copied (needed for the drift tab).

If you see **“No runs found”**, the MLflow server was not running or no training script completed successfully.

### 4. Test the REST API

**Option A — Docker (recommended)**

```bash
cd /path/to/assignment2/docker
docker compose up --build
```

Wait until the server is listening. Then:

```bash
curl -s "http://localhost:5002/api/purchase?customer_id=11000&product_id=771" | python -m json.tool
```

**If `localhost:5002` does not connect** (typical in a **dev container** where `127.0.0.1` is not the Docker host), from the **`assignment2/`** root run the helper (tries host **:5002**, then falls back to **`docker compose exec`** on in-container **:5000**):

```bash
cd /path/to/assignment2
bash docker/smoke_api.sh
```

**Check:** JSON includes `product_id`, `product_name`, `kind`, `score` (P(purchase)), `threshold`, `predicted_purchase`, and `cold_start_user` (true if the user id was unseen in the training user index; the row is still scored with default feature values). Example pair `(11000, 771)` is in `data/train_events.csv`.

- **Swagger UI:** open **http://localhost:5002/ui** or **http://localhost:5002/apidocs** (exact path depends on your Connexion version; if one 404s, try the other or read the container log for the docs URL).
- **Cold start:** use a `customer_id` that does not appear in training — you should get `cold_start_user: true`, a numeric `score` from default feature values, and a short `message`.

**From inside a Linux dev container** where `localhost` is not the host, try **`http://172.17.0.1:5002`** instead of `localhost`.

For **Postman**, **`socat` + port forwarding**, and copy-paste **`curl`** for each environment, see **Step 4 → Postman and terminal checks** (later in this file).

**Option B — Run API without Docker** (same code, local venv with `connexion` installed):

```bash
cd /path/to/assignment2/docker/src
pip install -r ../requirements.txt
python server.py
```

Then use **`http://localhost:5000`** in curl (no `5002` unless you map it yourself).

### 5. Test the Streamlit app

```bash
cd /path/to/assignment2/webapp
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open **http://localhost:8501** (or your forwarded URL).

1. **Purchase prediction tab** — set API base to **`http://localhost:5002`** (or **`http://172.17.0.1:5002`** from a dev container), enter `customer_id` and `product_id`, click **Predict purchase**, confirm **score** / **threshold** / **predicted purchase** appear.
2. **Drift tab** — after at least one recommendation request, confirm **`production_log.csv`** exists next to `app.py`. Use **“Seed demo log”** if the log is empty; then confirm KS results show for columns like **`CustomerID`** (requires overlapping columns between `raw_data.csv` and the log).

### 6. Optional: quick `purchase_prediction` import (no MySQL)

```bash
cd /path/to/assignment2/src
PYTHONPATH=. python -c "from purchase_prediction import EXPERIMENT_KIND, predict_purchase_for_pair; print('ok', len(EXPERIMENT_KIND))"
```

---

## Step 1: Prepare the data

```bash
cd /path/to/assignment2/src
python data_prep.py
```

Writes under **`../data/`**:

- `interactions_train.csv`, `interactions_test.csv` — line-level splits  
- `raw_data.csv` — train+test interactions (drift reference)  
- `recsys_meta.pkl` — id maps, test pairs, train user item sets, popularity  
- `training_stats.json`, `feature_names.json`

---

## Step 2: Train models (MLflow)

```bash
cd /path/to/assignment2/src
python baseline_model.py
python alternative_model_1.py
python alternative_model_2.py
python alternative_model_3.py
```

Each script builds (or reuses) **`data/train_events.csv`** and **`data/test_events.csv`**, calls **`prepare_interaction_split()`** for **meta** + **train_csr**, and logs **accuracy, precision, recall, f1** (and threshold params) to MLflow.

---

## Step 3: Export best model + seed the web app

```bash
cd /path/to/assignment2/src
python evaluate_models.py
```

Selects the experiment with the **highest f1**, refits the **winner** (tabular), writes **`docker/model/`** ( **`serve_bundle.pkl`**, **`recsys_meta.pkl`**, **`model_card.json`** ), and copies into **`webapp/`**:

- `webapp/training_stats.json`
- `webapp/raw_data.csv`

---

## Step 4: Run the prediction API (Docker)

From **`assignment2/docker/`** (compose build context is the **parent** `assignment2/` folder):

```bash
docker compose up --build
```

- **Host URL:** `http://localhost:5002` (maps host **5002** → container **5000**). **5002** avoids a common clash on **Mac** where **Cursor** may already use **5001** on the host.
- **Swagger UI:** Connexion (path depends on version; often `/ui` or `/apidocs`).
- **Route:** **`GET /api/purchase?customer_id=...&product_id=...`**

### Postman and terminal checks

Use **`GET`** for both endpoints. Base path on the **published** port is always **`5002`** on the machine where Compose bound it (host), and **`5000` only inside** the `api` container.

| Endpoint | Example URL (replace `<HOST>`) |
|----------|--------------------------------|
| Health | `http://<HOST>:5002/api/health` → `{"status":"ok"}` |
| Purchase | `http://<HOST>:5002/api/purchase?customer_id=11000&product_id=771` |

**Postman:** method **GET**. For purchase, either paste the full URL with query string or set URL to `http://127.0.0.1:5002/api/purchase` and add **Params**: **`customer_id`**, **`product_id`** (both required). Use **`127.0.0.1`** only when that host is really where port **5002** is published (same rules as the **`curl`** examples below).

**Postman Web vs Desktop:** the **browser-based Postman** often cannot call **`localhost`** / **`127.0.0.1`** on your machine unless the **[Postman Agent](https://learning.postman.com/docs/getting-started/installation/installation-and-updates/#postman-agent)** is running locally. For a **Docker API on your laptop**, **Postman Desktop** (or Desktop + Agent) is the usual approach and still satisfies assignment wording (“test with Postman”).

**`curl` (same endpoints; choose `<HOST>` to match your environment):**

```bash
# Same machine as Docker (e.g. MacBook + Docker Desktop) — port 5002 on localhost:
curl -sS "http://127.0.0.1:5002/api/health"
curl -sS "http://127.0.0.1:5002/api/purchase?customer_id=11000&product_id=771"

# Linux dev container: published 5002 is on the Docker host, not this shell's 127.0.0.1:
curl -sS "http://$(ip route | awk '/default/ {print $3}'):5002/api/health"
curl -sS "http://$(ip route | awk '/default/ {print $3}'):5002/api/purchase?customer_id=11000&product_id=771"

# Inside the api container (slim image has no curl — use Python; app port is 5000):
cd assignment2/docker
docker compose exec api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/api/health').read().decode())"
docker compose exec api python -c 'import urllib.request; u="http://127.0.0.1:5000/api/purchase?customer_id=11000&product_id=771"; print(urllib.request.urlopen(u).read().decode())'
```

**Postman on your laptop + Docker only on a remote Cursor/VS Code dev container:** add **port 5002** in the **Ports** / forwarding panel. The tunnel ends on **remote** `localhost:5002`. If **`curl` on the laptop** gives **empty reply** while the **gateway** URL works in the remote terminal, the API is on the Docker host but **not** on remote `127.0.0.1:5002`. Run a **TCP relay in the remote dev container** (own terminal tab, leave running):

```bash
sudo apt-get update && sudo apt-get install -y socat   # once, if socat is missing
socat TCP-LISTEN:5002,fork,reuseaddr TCP:$(ip route | awk '/default/ {print $3}'):5002
```

Then **`http://127.0.0.1:5002/api/health`** on the **laptop** (and the same base URL in Postman) should work while the forward and `socat` stay up.

The container copies **`src/purchase_prediction.py`** from the repo into `/app/src/` with **`clf_modeling`** for inference row building.

### Connection refused or `ERR_EMPTY_RESPONSE` in the browser

**1. Use the mapped host port (5002), not 5000.**  
Compose publishes **`5002` on the host → `5000` inside the container**. Opening **`http://localhost:5000/`** on your laptop hits **whatever** (if anything) is on host port **5000**, not the API — you will often see **ERR_EMPTY_RESPONSE** or random failures. Use **`http://127.0.0.1:5002/`** or **`http://127.0.0.1:5002/api/health`**.

**Browser:** Avoid **`http://host.docker.internal:5002/`** in the address bar on typical **Linux**: that name is often **not in DNS** (`DNS_PROBE_FINISHED_NXDOMAIN`). Docker Desktop on Mac/Windows usually defines it; Linux Docker Engine often does **not** (unless you add it yourself). For the browser, use **`http://127.0.0.1:5002/`** on the **machine where the port is published**, or the **forwarded** URL from Cursor/VS Code **Ports**, not `host.docker.internal`.

**2. `curl` from inside a dev container often cannot use `127.0.0.1:5002`.**  
When Docker runs **on the host** but your shell is **inside** the dev container, **localhost in that shell is the container itself**, not the machine where port **5002** was published. Then:

```text
curl: (7) Failed to connect to 127.0.0.1 port 5002 ... Couldn't connect to server
```

is expected. Do one of the following:

- Run **`curl`** from a terminal on the **same OS as Docker Desktop** (your **laptop** terminal), or  
- **Forward port 5002** in Cursor/VS Code from the remote session and open the **forwarded** URL in the browser, or  
- From **inside another container** (e.g. your dev container), **`127.0.0.1:5002` is still wrong** — published ports are on the **Docker host**. Try reaching the host first:

  ```bash
  # Often works on Docker Desktop / recent Docker Engine:
  curl -sS "http://host.docker.internal:5002/api/health"

  # On Linux, prefer the default gateway when host.docker.internal misbehaves (see note below):
  curl -sS "http://$(ip route | awk '/default/ {print $3}'):5002/api/health"
  ```

  **Note:** On some Linux setups, `host.docker.internal` exists but **`curl: (52) Empty reply from server`** — something on that path accepts the TCP connection but is not your Compose-published port. In that case the **`ip route` / default-gateway** URL is the one that actually reaches **`5002`** on the Docker host.

  One of these should match where **`0.0.0.0:5002`** was bound when Compose started the stack.  
- Test **inside** the API container (proves the app is listening). The API image is **slim** and usually has **no `curl`**; use Python:

  ```bash
  cd assignment2/docker
  docker compose exec api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/api/health').read().decode())"
  ```

  (Inside the `api` container the app listens on **5000**; the **host** uses **5002**.)

**3. Purchase route only after health works.**  
If **`/api/health`** returns `{"status":"ok"}` from `exec` but the browser still fails, the problem is **host ↔ Docker port publishing or port forwarding**, not Python. If **`/api/purchase`** errors, check **`docker compose logs api`** and that **`docker/model/serve_bundle.pkl`** exists (run **`evaluate_models.py`**).

**4. `curl: (52) Empty reply from server` or Postman “socket hang up” on the Mac, but `docker compose exec api` to `http://127.0.0.1:5000/...` works**  
The API is running **inside** the container. Traffic from **your Mac to `http://127.0.0.1:5002/...`** goes through Docker’s **host port map**; if TCP connects but you get **no JSON**, the process in the container may not be listening in a way the proxy accepts (a known quirk with the **Werkzeug** dev server and **Docker Desktop** on Mac), or the image is **stale** (old `CMD`).

Retrace in order:

1. **`docker compose down`** then **`docker compose up --build`** from `assignment2/docker` so the image picks up the current **Dockerfile** (API is started with **Gunicorn** on `0.0.0.0:5000`, not the toy dev server only).
2. On the Mac: **`docker ps`** — confirm **`0.0.0.0:5002->5000/tcp`** on the `api` container.
3. **`curl -sS "http://127.0.0.1:5002/api/health"`** on the **Mac** (not inside a dev container). Expect **`{"status":"ok"}`**.
4. If it still fails: **`docker logs <api-container-name> --tail 80`** while repeating the curl; look for Gunicorn boot lines and tracebacks. **`lsof -nP -iTCP:5002 -sTCP:LISTEN`** on the Mac should show **Docker** (or `com.docke`) owning the port, not another app.

**5. Confusing “where am I running curl?”**  
- **Mac + Docker Desktop:** use **`http://127.0.0.1:5002`**.  
- **Shell inside a dev container (not the Mac):** `127.0.0.1:5002` is **wrong**; use the **gateway** URL from **`smoke_api.sh`** or `http://172.17.0.1:5002` (see §**2** above).  
The guide is written so **step 4’s curl to `127.0.0.1:5002`** is for the **same machine that runs Docker Desktop**, matching Postman on that Mac.

---

## Step 5: Streamlit web app

### 5.1 Install dependencies

```bash
cd /path/to/assignment2/webapp
pip install -r requirements.txt
```

### 5.2 Data files beside `app.py`

| File | Purpose |
|------|---------|
| **`raw_data.csv`** | Drift reference (interaction columns). |
| **`training_stats.json`** | Optional JSON in expander. |
| **`production_log.csv`** | Appended per purchase prediction request. |

### 5.3 API base URL

Default **`http://172.17.0.1:5002`** for Streamlit inside a Linux dev container; use **`http://localhost:5002`** on the host. Override with **`PREDICT_API_BASE`**.

### 5.4 Start Streamlit

```bash
cd /path/to/assignment2/webapp
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### 5.5 Using the app

- **Purchase prediction tab:** `customer_id`, `product_id` → **`GET /api/purchase`** → **score / threshold / predicted_purchase** for that pair. Unknown **ProductID** returns **400**. Cold-start users get **P(purchase)** with default feature values (see `message`).
- **Drift tab:** KS tests on numeric columns present in both **`raw_data.csv`** and **`production_log.csv`** (e.g. `CustomerID`).

---

## Quick reference: ports and paths

| Service | Default | Notes |
|---------|---------|--------|
| MLflow | `http://localhost:9080` *or* `file:…/mlruns` | **Default** in code: **http://127.0.0.1:9080**; override with **`MLFLOW_TRACKING_URI`**. |
| API (Docker) | `http://localhost:5002` | `compose.yaml`: `5002:5000`. |
| Streamlit | `http://localhost:8501` | `--server.port` to change. |
| Model artifacts | `assignment2/docker/model/` | Filled by `evaluate_models.py`. |

---

## Quick start checklist

See **How to test everything (end-to-end)** above for detailed checks. Short version:

1. Start **MySQL** (or rely on **`data/raw_data.csv`**); start **MLflow** on **:9080** (or set **`MLFLOW_TRACKING_URI=file:…/mlruns`**); activate venv with **`lightgbm`**, **`xgboost`**, **`scikit-learn`**, etc. (see install lines above).
2. `cd assignment2/src && python data_prep.py`
3. Run the four training scripts; confirm metrics in MLflow.
4. `python evaluate_models.py` — confirm `docker/model/serve_bundle.pkl` exists.
5. `cd assignment2/docker && docker compose up --build` — curl `/api/purchase` (or `bash docker/smoke_api.sh` from `assignment2/` if `localhost:5002` fails).
6. `cd assignment2/webapp && streamlit run app.py` — exercise Purchase prediction + Drift tabs.

---

## Notes

- **Task A** (`docs/TASK_A_problem_formulation.md`): **binary** **purchase** classification on **event** rows; **F1** is the primary **offline** selection metric, with **accuracy / precision / recall** logged; the **API** exposes **per (customer, product) purchase** prediction using the **winner** model (**P(purchase)** + threshold).
- **Evaluation:** temporal **per-customer** holdout on lines; **test** **event** **table** includes **sampled** negatives (see `classification_data.py`).
- **Docker image** installs **Connexion**, **scikit-learn**, **xgboost**, **lightgbm**, and related scientific Python wheels (no PyTorch in the default API build).
