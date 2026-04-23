# Task A — Problem formulation (BT4301 Assignment 2)

## Predictive analytics business question

**Operational predictive question (this assignment):** For a given **(customer, product)** in the historical log, can we **learn** whether that pair is a **purchase (positive)** or **not a purchase in our label design (negative)**? In other words, the models are trained and evaluated on **supervised** **binary** labels **Y ∈ {0, 1}** that indicate **bought vs not** for that **row** (not a live experiment “after we show a recommendation, will they check out”).

**Business framing (recommendation):** The store still needs **relevant product recommendations**; classifiers produce **purchase propensity** scores for **(customer, product)** pairs. The business goal remains “who should see which products,” and the **stated** **ML task** in the implemented pipeline is **binary classification** on **(customer, product)** **events** with **sampled** negatives, not a pure recall@K-only benchmark.

**Implementation summary:** A **per-customer temporal 80/20** split on sales lines. **Train/test event tables** include **enriched** product, customer, and time features. **Training:** **five** **sampled** **negative (customer, product) rows per user** per split (pooled from the train product set; negatives = “not bought” under that design). **Models:** **logistic regression** (baseline), **random forest**, **XGBoost**, and **LightGBM** (all on the same tabular event features). **Model selection (primary):** **F1** on the **test** event table, with a **tuned** probability **threshold** for each model, alongside **accuracy, precision, and recall**.

### Dependent variable and pattern (solution statement)

- **Unit of analysis:** **Customer–product** rows in the event table, with **OrderDate**-based time features and train-only **aggregates** where applicable.
- **Target Y = 1:** A **real purchase line** in the **train** or **test** window (per the temporal split).  
- **Target Y = 0:** A **(customer, product)** pair that **did not** occur as a purchase among that user’s **labeling** rules: **sampled** from products available in the **train** pool, excluding the user’s **train** purchases (and, for the **test** set, also excluding that user’s **test** purchase lines when sampling).
- **Pattern type:** **Predictive** **binary classification** on tabular **event** features (interaction data also feeds `prepare_interaction_split` for catalog metadata, not a separate matrix-only model at serve time). **Not** the main focus: **regression** on **quantity/price** as a target, **customer segmentation** as the **deliverable**, or **association rules** as the **sole** method.

### How the four models map to the task

| Model | Role |
|--------|------|
| **Logistic regression** | **Baseline** **tabular** classifier: calibrated **purchase propensity** from encoded categoricals + scaled numerics. |
| **Random forest** | **Nonlinear** **tabular** classifier on the same event features. |
| **XGBoost** | **Gradient-boosted** trees on the same event features; **class imbalance** handled via **scale_pos_weight** where used. |
| **LightGBM** | **Gradient-boosted** trees on the same event features; class imbalance via **scale_pos_weight** in line with XGBoost. |

**Caveat on “will be purchased”:** The label is **“was purchased in the split”** (or **is** a **sampled** **non-purchase** row), not a forward **randomized** **treatment** (“after showing product X, purchase within 7 days”). That is still standard **supervised** **predictive** **analytics** on **log** data.

---

## Analytics approach

1. **Data:** Line-level fact rows (**`CustomerID`**, **`ProductID`**, **`OrderDate`**, etc.) and joins to **product** / **customer** for features; **temporal** split per `data_prep` / `classification_data`.
2. **Models (MLflow):** As in the table above; metrics logged (at minimum) **accuracy, precision, recall, F1**; **best** run chosen by **F1** in **`evaluate_models.py`**.
3. **Serving:** **`docker` API** exposes **`GET /api/purchase`** for **(customer_id, product_id)** **pair** prediction (see **REST web service and Docker deployment** below).

---

## REST web service and Docker deployment

The assignment asks to **deploy the best model via a REST endpoint in a Docker container**, either by **packaging the pickle of the best model from MLflow into the image** or by **loading the model from MLflow at runtime**. This implementation uses the **first** option (and does **not** call MLflow during container inference).

### Role of MLflow (selection only, not runtime serving)

- **`src/evaluate_models.py`** uses the **MLflow Tracking API** (`MlflowClient`) to read the **latest run** in each of the four experiments (baseline **logistic regression**, **RF**, **XGBoost**, **LightGBM**), compares **logged `f1`** on the **test event** table, and picks the experiment with the **highest F1**.
- The **winning run id** and comparison are written to **`docker/model/model_card.json`** for traceability. **No model binary is downloaded from MLflow** in this step; the script only reads **metrics and metadata** from the tracking store (with **`MLFLOW_TRACKING_URI`** / default URI as in **`mlflow_util.py`**).

### Building the deployable bundle (refit, then package)

After selection, the script **re-fits** the **winner** on **full** training data (as required for a clean production model), **not** by pulling an artifact from MLflow:

- **Tabular (log_reg / rf / xgb / lgbm):** `fit_clf` trains a new pipeline; **`clf_extras`**, **`clf_threshold`**, and **meta** are included in a Python **`dict`** (saved in **`serve_bundle.pkl`**).

The full inference bundle is saved with **`joblib.dump`** to **`docker/model/serve_bundle.pkl`** (plus **`recsys_meta.pkl`** for metadata). This file is the **“pickle packaged into the container”** for deployment.

### Docker image and REST API

- The **Dockerfile** copies **`docker/model/`** (including **`serve_bundle.pkl`**) and **`docker/src/`** (Connexion app, OpenAPI spec, `predict` handler) into the image at **build** time. The container does **not** need network access to MLflow to **answer requests**.
- **Connexion** serves **`GET /api/purchase`** (and **`/api/health`**) on port **5000** inside the container; **`docker/compose.yaml`** maps host port **5002** → **5000**.
- At first request, **`docker/src/predict.py`** **lazy-loads** **`serve_bundle.pkl`**, then calls **`predict_purchase_for_pair`** in **`purchase_prediction.py`** to return **score**, **threshold**, and **predicted_purchase** for the pair.

### Why not “retrieve from MLflow at runtime” (optional design note)

A **runtime** design could **`mlflow.pyfunc.load_model` / registry URI** on startup. The **packaged** approach was chosen so that **inference** depends only on the **image + bundle**, with **reproducible** artifacts and **no** production dependency on a **live** MLflow server. Regenerating the image after **`evaluate_models.py`** is the supported workflow if the **best** model or **data** change.

---

## Relevant tables (star schema)

- **`sales`:** `CustomerID`, `ProductID`, `SalesOrderID`, optional `LineTotal` / `OrderQty` for weighting; line identity.
- **`salesordertime`:** **`OrderDate`** for chronological ordering per customer.
- **`product`:** Attributes (e.g. **category, list price, color, model**).
- **`customer`:** Address / geography (e.g. **city, region, country**).

---

## Scope and limitations

- **Label imbalance:** Many **Y = 1** **lines** per user vs only **five** **Y = 0** **rows** per user in the event table; **class weights** / **tuned** **thresholds** are used. Results depend on the **sampling** rule for **0s**.
- **Cold-start** users: customers **not** in the **train** **user** **index** are still scored with **default** demographic / aggregate features. See **`predict_purchase_for_pair`** in **`purchase_prediction.py`**.
- **Temporal leakage:** **Features** and **splits** follow **`classification_data`**: **test** **negatives** are **not** drawn from **purchased** **products** in **train** for that user (per implementation).

---

## Deliverables alignment

- **`src/data_prep.py`**, **`src/classification_data.py`:** **Interactions** and **event** **tables**; optional **`data/train_events.csv`**, **`test_events.csv`**, **`clf_extras`**, etc.
- **`src/baseline_model.py`**, **`alternative_model_1.py`**, **`alternative_model_2.py`**, **`alternative_model_3.py`:** train and **log** **metrics**; **`src/evaluate_models.py`** **exports** the **F1**-best bundle to **`docker/model/`** and copies **reference** **CSVs** for **Streamlit** / drift as **before** (see **`IMPLEMENTATION_MAP.md`**, **`guide.md`**).

---

## Task D — Web application, monitoring, and feedback loop (implementation)

This section ties **Task D** to the business question above: a **merchandising / offer** use case in which the store (or a downstream system) must **decide** whether a **(customer, product)** **pair** is worth **prioritising** (e.g. in a list or a campaign). The **Streamlit** app in **`webapp/app.py`** **consumes** the **Task C** **REST** service to **operationalise** that **decision** and implements a **monitoring and feedback** view on a **single** dashboard, satisfying **Task D**’s “simple web application + dashboard” requirement.

### Business process automated by the web app (Task C integration)

- **User action:** The analyst enters a **base URL** for the **Docker** **API** (e.g. **`http://localhost:5002`**) and a **(customer_id, product_id)** pair, then runs **“Predict purchase”.**
- **Automation:** The app calls **`GET /api/purchase`** (same **Connexion** contract as Task C) and **displays** the **model output** in plain language: **model type** (e.g. XGBoost / LightGBM), **P(purchase)** (shown to **two** decimal places), **decision cutoff**, and a **verdict** (likely vs unlikely to purchase). That **replaces** a manual **ad hoc** process (e.g. pulling facts from the warehouse) with a **single** **governed** **interface** to the **serving** model.
- **Post-production log:** Every successful prediction appends a **fixed-schema** row to **`webapp/input_monitoring_log.csv`**, so **production traffic** is available for the monitoring tab without a separate ETL job.

The assignment’s **“either (1) ground truth (2) input drift”** is addressed with **both** a **ground-truth** **interpretation** and **univariate** **input** **drift** **detection** (both **visible** in the same **dashboard**).

### (1) Ground truth evaluation (post-production context)

- **Source of “truth” here:** The **Adventure Works** **line** **facts** in **`raw_data.csv`** (copied beside the app from **`evaluate_models.py`**) represent **historical** **(customer, product)** **purchases** (order lines) used as the **drift** **reference** and as a **static** **label** for “**has this** **pair** **ever** **appeared** as a **purchase** in our **export**.”
- **What the dashboard shows:** For each row in the **production log**, the app checks whether the **(CustomerID, ProductID)** **appears** in that **reference** set. The **“Ground truth (user × product history)”** block reports **per-row** text: either a **ground-truth** **positive** (user **bought** this product **in** the **reference** **history**) or **“User has not bought this item before”** in the **reference** data, plus **summary** **counts** across logged requests.
- **Metrics:** These are **operational** **transparency** **metrics** (share of **requests** for **previously** **seen** **pairs** vs **novel** **pairs**), not a full **out-of-sample** **F1** on **new** **labels**—but they are **genuine** **post-production** **context** tied to the **Task A** **buy / not** **semantics** on **logged** **pairs**. Where **labels** for **outcomes** of **recommendations** are **unavailable** in a **toy** **deployment**, this **is** a **practical** **“feedback”** **signal** the business can **read** on the **same** **screen** as the **drift** **triage**.

### (2) Input drift detection (univariate statistical tests)

- **Reference vs production:** **“Training** **reference**” = **`raw_data.csv`**; **“production** **inputs**” = **`input_monitoring_log.csv`** (optional **synthetic** **runs** for reports are supported by **`webapp/simulate_drift.py`**).
- **Tests:** For each **numeric** **column** in **`CustomerID`**, **`ProductID`**, **`SalesOrderID`**, **`LineTotal`** that exists in **both** tables, with **enough** **non-missing** **values** on **each** **side** (minimums align with the **code** in **`app.py`**), the app runs a **two-sample** **Kolmogorov–Smirnov** (KS) **test** and records **p-value** and a **per-feature** **drift** **flag** (**p** **<** **0.05**).
- **Feedback loop (traffic light):** The app computes the **fraction** of **tests** that **flag** **drift** and maps that to a **green** / **yellow** / **red** **recommendation** (with **retraining** **called** **out** on **widespread** **shift**), matching **Task** **D**’s ask for a **“simple** **data** **dashboard**” with **univariate** **test** **results** and a **plain** **action** **prompt**.

### How this answers Task D in one place

- **Web stack:** **Streamlit** **Python** app; **no** **separate** **front-end** **framework** **required** by the **brief**.
- **Task C in the loop:** **Same** **REST** **contract**; **automation** of **(customer, product)** **scoring** for the **stated** **Task** **A** **problem** (propensity to **treat** a **line** as a **purchase**).
- **Monitoring:** **Input** **drift** (KS) **and** a **ground-truth** **/** **history** **panel** in **one** **dashboard**; **production** **inputs** are **logged** to **support** both **slices**.

For **repro** and **submissions**: run **`cd`** **`webapp`** then **`streamlit`** **`run`** **`app.py`**, and point the app at a **live** **Task** **C** **base** **URL**; for **staged** **drift** **examples**, see **`webapp/simulate_drift.py`** and **`guide.md`**.

