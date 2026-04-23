# Service Guide (Docker Container)

Since this Ubuntu environment runs inside a Docker container without systemd,
use init.d scripts or manual commands instead of `systemctl`.

---

# Jenkins


## Log in to Jenkins

```bash
username: admin
password: admin123
```


## Start Jenkins

```bash
/etc/init.d/jenkins start
```

## Stop Jenkins

```bash
/etc/init.d/jenkins stop
```

## Check Status

```bash
/etc/init.d/jenkins status
```

Or check for a running process:

```bash
ps aux | grep jenkins
```

## View Logs

Follow the log in real time (Ctrl+C to stop watching):

```bash
tail -f /var/log/jenkins/jenkins.log
```

## Access Jenkins

Open in browser: http://localhost:8080

Initial admin password (first-time setup only):

```bash
cat /var/lib/jenkins/secrets/initialAdminPassword
```

---

# MySQL

## Start MySQL

```bash
mysqld_safe &
```

Or:

```bash
/etc/init.d/mysql start
```

## Stop MySQL

```bash
/etc/init.d/mysql stop
```

Or:

```bash
mysqladmin -u root shutdown
```

## Check Status

```bash
/etc/init.d/mysql status
```

## Connect

```bash
mysql -u root
```

## View Logs

```bash
tail -f /var/log/mysql/error.log
```

## Key Paths

- Config: `/etc/mysql/mysql.cnf`
- Data: `/var/lib/mysql/`
- Error log: `/var/log/mysql/error.log`
- Socket: `/var/run/mysqld/mysqld.sock`

## Log in for the default account:
```bash
user     = debian-sys-maint
password = BlfwG1dkOTkIvbJo
```

---

# Apache Airflow

## Environment Variable

Already set in `~/.bashrc`:

```bash
export AIRFLOW_HOME=/root/airflow
```

If opening a new shell and it's not set, run:

```bash
source ~/.bashrc
```

## Initialize Database (first time only)

```bash
source ~/venv/bin/activate
airflow db migrate
```

## Start Airflow Webserver

```bash
source ~/venv/bin/activate
airflow webserver --port 8081 &
```

(Use port 8081 to avoid conflict with Jenkins on 8080)

## Start Airflow Scheduler

```bash
source ~/venv/bin/activate
airflow scheduler &
airflow dag-processor &
airflow api-server
```

## Stop Airflow

```bash
pkill -f "airflow webserver"
pkill -f "airflow scheduler"
```

## Log in to Airflow

```
username: admin
password: password (wXF44qEx7YuwbgGw)
```

## Access Airflow

Open in browser: http://localhost:8081

## Key Paths

- Config: `/root/airflow/airflow.cfg`
- DAGs: `/root/airflow/dags/`
- Logs: `/root/airflow/logs/`

---

# Hadoop HDFS

Hadoop Distributed File System — a distributed, fault-tolerant file system
for storing large datasets across multiple nodes. In this single-node setup
it runs locally but behaves the same as a multi-node cluster.

## Start HDFS

```bash
start-dfs.sh
```

This starts three processes:
- **NameNode** — manages the filesystem namespace (tracks which files exist and where blocks are stored)
- **DataNode** — stores the actual data blocks
- **SecondaryNameNode** — periodically checkpoints the NameNode metadata (not a failover node)

## Stop HDFS

```bash
stop-dfs.sh
```

## Check Status

```bash
jps
```

You should see: NameNode, DataNode, SecondaryNameNode.

## Common Commands

```bash
hdfs dfs -ls /                     # list root directory
hdfs dfs -mkdir /mydir             # create a directory
hdfs dfs -put localfile /mydir/    # upload a file
hdfs dfs -get /mydir/file .        # download a file
hdfs dfs -rm /mydir/file           # delete a file
```

## Web UI

- NameNode: http://localhost:9870

## Key Paths

- Install: `/root/hadoop/`
- Config: `/root/hadoop/etc/hadoop/`
- Data: `/root/hadoopdata/namenode/` and `/root/hadoopdata/datanode/`
- Logs: `/root/hadoop/logs/`

---

# Hadoop YARN

Yet Another Resource Negotiator — a cluster resource manager that schedules
and monitors jobs (MapReduce, Spark, etc.) running on the cluster. HDFS stores
the data; YARN manages the compute.

## Start YARN

```bash
start-yarn.sh
```

This starts two processes:
- **ResourceManager** — accepts job submissions and allocates cluster resources
- **NodeManager** — runs on each node, manages containers and reports resource usage

## Stop YARN

```bash
stop-yarn.sh
```

## Check Status

```bash
jps
```

You should see: ResourceManager, NodeManager (in addition to HDFS processes).

## Web UI

- ResourceManager: http://localhost:8088

## Start/Stop Everything (HDFS + YARN)

```bash
start-dfs.sh && start-yarn.sh     # start all
stop-yarn.sh && stop-dfs.sh       # stop all
```

After starting both, `jps` should show:
NameNode, DataNode, SecondaryNameNode, ResourceManager, NodeManager.

> If you also need Hive, see the **Apache Hive** section below for the
> full start/stop sequence including HiveServer2 and Beeline.

---

# Apache Kafka

## Start Kafka

```bash
cd /root/kafka/kafka_2.13-4.2.0
bin/kafka-server-start.sh config/server.properties
```

Kafka runs on port 9092 by default.

## Stop Kafka

Press **Ctrl+C** in the terminal where Kafka is running.

## Key Paths

- Install: `/root/kafka/kafka_2.13-4.2.0/`
- Config: `/root/kafka/kafka_2.13-4.2.0/config/server.properties`

---

# Apache Hive

HDFS and YARN **must** be running before Hive can start.
Derby (embedded metastore) only allows one process at a time, so run
HiveServer2 alone -- it embeds the metastore internally.

## Full Start Sequence (HDFS + YARN + Hive)

```bash
# 1. Start HDFS
start-dfs.sh

# 2. Start YARN
start-yarn.sh

# 3. Verify HDFS and YARN are up
jps
# Expected: NameNode, DataNode, SecondaryNameNode, ResourceManager, NodeManager

# 4. Start HiveServer2 (embeds metastore, opens ports 10000 & 10002)
hiveserver2 &

# 5. Wait ~30 seconds for HiveServer2 to initialise, then connect
beeline -u jdbc:hive2://localhost:10000 -n root
```

## Full Stop Sequence (Hive + YARN + HDFS)

```bash
# 1. Exit Beeline (if connected)
!quit

# 2. Stop HiveServer2
pkill -f "proc_hiveserver2"

# 3. Stop YARN
stop-yarn.sh

# 4. Stop HDFS
stop-dfs.sh
```

## Check Status

```bash
jps
```

With everything running you should see:
NameNode, DataNode, SecondaryNameNode, ResourceManager, NodeManager,
and a RunJar process (HiveServer2).

## Connect with Beeline

```bash
beeline -u jdbc:hive2://localhost:10000 -n root
```

Exit Beeline:

```sql
!quit
```

## Access

- HiveServer2 Thrift port: `10000`
- HiveServer2 Web UI: http://localhost:10002

## Key Paths

- Install: `/root/hive/`
- Config: `/root/hive/conf/hive-site.xml`
- Metastore DB (Derby): `/root/hive/metastore_db/`
- Warehouse (HDFS): `/user/hive/warehouse`

## Re-initialise Metastore (if needed)

Stop HiveServer2 first, then:

```bash
pkill -f "proc_hiveserver2"
rm -rf /root/hive/metastore_db
schematool -dbType derby -initSchema
```

## Troubleshooting

- **Derby lock error:** A stale lock file can remain after a crash. Fix:
  `rm -f /root/hive/metastore_db/db.lck /root/hive/metastore_db/dbex.lck`
- **"User root is not allowed to impersonate root":** Ensure `core-site.xml`
  contains `hadoop.proxyuser.root.hosts = *` and `hadoop.proxyuser.root.groups = *`,
  then restart HDFS/YARN.
- **HiveServer2 won't start (PID file exists):**
  `rm -f /root/hive/conf/hiveserver2.pid`

---

# Python / Jupyter (BT4301 data warehouse notebooks)

## pandas `to_sql` and SQLAlchemy

**pandas 2.3.x** only treats SQLAlchemy as available if **SQLAlchemy >= 2.0** is installed. If you use SQLAlchemy 1.4.x, `DataFrame.to_sql(..., con=engine)` is mis-routed through pandas’ SQLite backend and fails with `AttributeError: 'Engine' object has no attribute 'cursor'`.

**Fix (in the same venv as the notebook kernel):**

```bash
/root/venv/bin/pip install 'SQLAlchemy>=2.0.0'
```

Then use either `con=engine` or `with engine.begin() as conn: df.to_sql(..., con=conn)`.

**Note:** Upgrading SQLAlchemy to 2.x can conflict with packages that pin `SQLAlchemy<2` (e.g. some Airflow / Flask-AppBuilder stacks). For course work, use a **dedicated venv** for BT4301 if those tools share the same environment.

---

# BT4301 Assignment 2 (product recommendations)

Paths under `/root/assignment2/`. Task: **implicit-feedback top‑K recommendations** (user CF, item CF, ALS MF, NCF); metrics are **recall@10** / **precision@10**, not regression RMSE.

## Python deps (training venv)

Besides `mysql-connector-python`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `mlflow`, `joblib`, install:

```bash
/root/assignment2/.venv/bin/pip install 'implicit>=0.7' 'torch>=2.2'
```

Docker API image installs the same stack via `assignment2/docker/requirements.txt` (includes **PyTorch** + **implicit**).

## MLflow

**Start** (from any cwd; uses SQLite backend in the directory where the server was first started):

```bash
mlflow server --host 0.0.0.0 --port 9080
```

**Check:** open http://localhost:9080

## Train models and export Docker bundle

Requires MySQL `datawarehouse` populated from Assignment 1 ETL.

```bash
cd /root/assignment2/src
python3 data_prep.py
python3 baseline_model.py
python3 alternative_model_1.py
python3 alternative_model_2.py
python3 alternative_model_3.py
python3 evaluate_models.py
```

## Predict API (Docker)

**Build / start** (compose **context** is `assignment2/` parent; host port **5001** → container **5000**):

```bash
cd /root/assignment2/docker
docker compose up --build -d
```

**Check:**

```bash
curl "http://127.0.0.1:5001/api/predict?customer_id=11000&top_k=5"
```

From a **dev container** (Docker CLI talks to the host daemon), if that fails use:

```bash
curl "http://172.17.0.1:5001/api/predict?customer_id=11000&top_k=5"
```

**Stop:**

```bash
cd /root/assignment2/docker && docker compose down
```

**Key paths:** `assignment2/docker/model/serve_bundle.pkl`, `recsys_meta.pkl`, `model_card.json`; OpenAPI `assignment2/docker/src/predict.yml`; `assignment2/src/recsys_lib.py` is copied into the image at build time.

## Streamlit monitoring app

```bash
pip install -r /root/assignment2/webapp/requirements.txt
```

**Streamlit → API URL (pick one):**

- **Dev container + Docker socket** (API is a sibling container): the API is **not** on `127.0.0.1` inside that container. Use the Docker bridge gateway:

```bash
PREDICT_API_BASE=http://172.17.0.1:5001 streamlit run /root/assignment2/webapp/app.py --server.address 0.0.0.0 --server.port 8501
```

- **Docker only on your laptop** (no extra dev container): `http://127.0.0.1:5001` / `http://localhost:5001` is fine; you can omit `PREDICT_API_BASE` if you set it in the Streamlit sidebar.

```bash
streamlit run /root/assignment2/webapp/app.py --server.address 0.0.0.0 --server.port 8501
```

**Access:** Forward port **8501** in Cursor and open the forwarded `localhost` URL — **recommendations** form (calls REST API) and drift dashboard. Logs requests to `assignment2/webapp/production_log.csv`.

---

## Power down (BT4301 Assignment 2 stack)

Stop things in reverse order of how you started them.

### Predict API (Docker Compose)

```bash
cd /root/assignment2/docker && docker compose down
```

Optional: remove the image as well:

```bash
docker compose -f /root/assignment2/docker/compose.yaml down --rmi local
```

### Streamlit

In the terminal where it is running: **Ctrl+C**.

If it was started in the background:

```bash
pkill -f "streamlit run /root/assignment2/webapp/app.py" || true
```

### MLflow tracking server

In the terminal where it is running: **Ctrl+C**.

If it was started in the background:

```bash
pkill -f "mlflow server" || true
```

### MySQL (only if you started it for this work)

```bash
/etc/init.d/mysql stop
```

### Check nothing is still listening (optional)

```bash
docker ps
ss -tlnp | grep -E ':5001|:8501|:9080|:3306' || true
```

---

# hdb-price-estimator (cloned repo)

Repository path: **`/root/hdb-price-estimator`** — Singapore **HDB resale price** estimation: Airflow DataOps (ingest → clean → transform → train), **MySQL** `HDB_Data`, **MLflow** tracking, **FastAPI** inference, **Streamlit** map + estimator. Upstream: [github.com/hamy-nguyen/hdb-price-estimator](https://github.com/hamy-nguyen/hdb-price-estimator).

Full setup, DAG behaviour, and troubleshooting: **`/root/hdb-price-estimator/README.md`**. Architecture diagram and path reference: **`/root/hdb-price-estimator/PROJECT_ARCHITECTURE.md`**. API-only: **`/root/hdb-price-estimator/api/README.md`**.

## Layout (high level)

| Path | Role |
|------|------|
| `airflow/dags/` | DAGs: `ingest_dag.py` (`data_ingest`), `clean_dag.py` (`data_clean`), `transform_dag.py` (`data_transform`), `train_dag.py` (`data_train`); helpers under `helpers/` (extract/upsert, watermarking, cleaners, spatial joins). |
| `dataset/raw/` | Source CSVs (HDB, MRT, bus, POI, OneMap exports, etc.). |
| `dataset/processed/` | Cleaned derivatives (often gitignored). |
| `notebooks/` | EDA, feature engineering, baseline models (e.g. LR, ridge/DT, RF/XGBoost), carpark / OneMap / tourist attraction notebooks. |
| `scripts/` | `ml_transform.py`, `onemap_address_search.py` (OneMap / geospatial utilities). |
| `web_application/` | `streamlit.py` dashboard; `predict_api_params.py` builds payloads for the API. |
| `api/` | FastAPI app (`app/main.py`, `model.py`, `schemas.py`), `Dockerfile`, `models/` for `model.pkl`. |
| `start_airflow.sh` | Sets `AIRFLOW_HOME` to repo `airflow/`, activates `.venv`, starts scheduler + dag-processor + **api-server** on **8081**. |

## Prerequisites

- Python **3.11+**, **MySQL 8.0+**
- Project venv: `python -m venv .venv && source .venv/bin/activate` then `pip install -r requirements.txt` (root `requirements.txt` pins **pandas &lt; 2.2** for Airflow / SQLAlchemy compatibility).

## Database

Create DB and user (see project README for SQL). Example:

- Database: **`HDB_Data`**
- Airflow connection id: **`mysql_default`** → `mysql://bt4301:<password>@localhost:3306/HDB_Data`

## Pipeline (Airflow)

Chain after manual trigger of **`data_ingest`**:

`data_ingest` (@monthly) → `data_clean` → `data_transform` → `data_train`

- Ingest: data.gov.sg APIs + local CSVs → `raw_*`; incremental resale month ingest; **SHA-256 row fingerprints** (`_fp`) for integrity.
- Clean → `clean_*`; Transform → geospatial joins → **`transform_resale_flat_price`** (ML-ready table).
- Train: Linear / Ridge / XGBoost pipelines; **MLflow** experiment **`HDB Resale Price Prediction: Auto Training`**; best artefact saved to **`api/models/model.pkl`**; DAG can **POST `/reload-model`** on the API.

## Start / stop (this environment)

**`AIRFLOW_HOME`:** export to **`/root/hdb-price-estimator/airflow`** (or use `start_airflow.sh` from repo root; expects `.venv` there).

```bash
cd /root/hdb-price-estimator
export AIRFLOW_HOME=/root/hdb-price-estimator/airflow
source .venv/bin/activate   # use this venv only — not e.g. /root/venv
# First time: airflow db migrate, configure airflow.cfg per project README
./start_airflow.sh
```

**Troubleshooting:** If `airflow scheduler` / `airflow api-server` crashes with `MappedAnnotationError` / `TaskInstance.dag_model`, you are usually on the wrong Python env or an old Airflow (e.g. **3.0.0** in `/root/venv` with **SQLAlchemy 2.0.49**). Run Airflow from **`hdb-price-estimator/.venv`** (Airflow **3.2.x** here) or upgrade that other venv’s `apache-airflow` to match.

**Example DAGs still in the UI / `airflow dags list` fails:** After `[core] load_examples = False`, old rows for bundle **`example_dags`** can remain in **`airflow/airflow.db`** (dozens of tutorial DAGs; `TimetableNotRegistered` when deserializing). Use the DAG search box for **`data_ingest`**, or stop Airflow and remove those metadata rows (or run **`airflow db reset`** if you can wipe local Airflow state). You should then see only **`data_ingest`**, **`data_clean`**, **`data_transform`**, **`data_train`**.

**`sqlite3.OperationalError: database is locked` / `httpx.ReadTimeout` on tasks:** The bundled metadata DB is **SQLite** (`airflow/airflow.cfg` → `sql_alchemy_conn`). **LocalExecutor** with high parallelism makes many processes write to that file at once and it locks. This repo lowers **`parallelism`** / **`max_active_tasks_per_dag`** in `airflow.cfg` for local dev. Do not run a second copy of scheduler/api-server on the same `AIRFLOW_HOME`. For heavier parallelism, use **PostgreSQL** (or MySQL) for Airflow metadata instead of SQLite.

- **Airflow UI:** http://localhost:8081  
- **MLflow** (start separately): `python -m mlflow server --host 127.0.0.1 --port 9080` → http://localhost:9080  

### Hosted inference API (no local `uvicorn`)

If the FastAPI app is deployed in the cloud (e.g. **Hugging Face Spaces** via `.github/workflows/deploy-hf.yml`), you **do not** need to run the API on your machine. The Streamlit app reads the predict URL from **`PREDICT_API_URL`** (see `web_applicatiopass/predict_api_params.py`).

```bash
cd /root/hdb-price-estimator && source .venv/bin/activate
export PREDICT_API_URL="https://YOUR_SPACE_SUBDOMAIN.hf.space/predict"
streamlit run web_application/streamlit.py
```

Replace the URL with your Space’s public URL + **`/predict`** (check the Space “App” link; HF may show the base URL only — append `/predict` for `POST`).

**Still local:** Streamlit loads the map from **MySQL** (`transform_resale_flat_price` on `localhost`). So you need a running DB + pipeline data for the dashboard; only the **model inference** call goes to the cloud. Credentials come from repo **`.env`**: **`MYSQL_HOST`**, **`MYSQL_USER`**, **`MYSQL_PASSWORD`**, **`MYSQL_DATABASE`** (defaults match the project README user **`bt4301`** / **`HDB_Data`**).

**Airflow `data_train` hot-reload:** if the API is only in the cloud, set **`API_RELOAD_URL`** in the Airflow environment to your deployed **`POST /reload-model`** URL (or leave default and accept that reload may fail until the Space restarts / redeploys).

### Optional: local FastAPI (development)

Only if you want to run the API on this machine: you must start **`uvicorn` from the `api/` directory** (so Python can import `app`), not from the repo root — otherwise you get `ModuleNotFoundError: No module named 'app'`.

```bash
cd /root/hdb-price-estimator/api && source ../.venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

Endpoints: **`GET /health`**, **`POST /predict`**, **`POST /reload-model`**. Swagger: **`/docs`**. Without `model.pkl`, API runs in **dummy** mode (~$500k placeholder).

**Streamlit:**

```bash
cd /root/hdb-price-estimator && source .venv/bin/activate
# Omit PREDICT_API_URL to use default http://localhost:7860/predict, or set as above for cloud.
streamlit run web_application/streamlit.py
```

→ http://localhost:8501 (forward port in Cursor if remote). **OneMap:** `.env` at repo root with `ONEMAP_EMAIL` / `ONEMAP_EMAIL_PASSWORD` (see project README).

**Calling a local API from a dev container:** use **`http://172.17.0.1:<port>`** instead of `127.0.0.1` when appropriate (same idea as BT4301 Assignment 2).

## Docker / Hugging Face

- Build/run API: see **`api/README.md`** (`docker build`, `docker run`, mount `models/`).
- CI deploy to Hugging Face Spaces: **`.github/workflows/deploy-hf.yml`**; secret **`HF_TOKEN`**.

## Power down (hdb-price-estimator)

- Stop **Airflow**: terminal running `start_airflow.sh` → **Ctrl+C** (script kills scheduler/dag-processor); or `pkill -f "airflow scheduler"` / `pkill -f "airflow dag-processor"` / `pkill -f "airflow api-server"` as needed.
- Stop **uvicorn** / **Streamlit** / **MLflow**: **Ctrl+C** in those terminals, or matching `pkill -f`.
- **MySQL:** `/etc/init.d/mysql stop` if you only use it for this project and want it off.



cd /root/hdb-price-estimator
source .venv/bin/activate
export AIRFLOW_HOME=/root/hdb-price-estimator/airflow

# Remove stale entry if you added a wrong one before
airflow connections delete mysql_default 2>/dev/null || true

# Add MySQL (replace password if yours is not "password")
airflow connections add mysql_default --conn-uri "mysql://bt4301:password@localhost:3306/HDB_Data"



airflow scheduler &
airflow dag-processor &
airflow api-server