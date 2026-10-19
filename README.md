# Heart Disease Prediction MLOps Pipeline

An academic MLOps project that combines four UCI Heart Disease datasets, prepares data for binary classification, trains and evaluates scikit-learn models, tracks experiments with MLflow, and serves predictions through FastAPI. Docker and Kubernetes configurations package and run the API, while GitHub Actions automates linting, testing, training, artifact checks, and image builds.

> **Disclaimer:** This project is for educational and demonstration purposes only. Its predictions are not medical advice and must not be used to diagnose, treat, or make decisions about a person’s health.

## Project Overview

The project demonstrates a reproducible workflow across model development and API operations:

1. Combine the Cleveland, Hungarian, Switzerland, and VA UCI Heart Disease datasets into a unified CSV.
2. Inspect and prepare the data, including missing-value handling and target conversion.
3. Build a train/test split and fit preprocessing transformations on the training partition.
4. Train and compare Logistic Regression and Random Forest classifiers.
5. Log model runs, parameters, evaluation metrics, and artifacts to MLflow.
6. Serve predictions and health/monitoring endpoints with FastAPI.
7. Package the application in Docker and run it through Kubernetes manifests.
8. Run automated code-quality, test, model-training, artifact, and image-build checks in GitHub Actions.

## Problem Statement

Heart disease datasets contain clinical measurements that can be used to explore supervised classification. The source UCI datasets encode disease diagnosis using multiple values. For this project, the prediction task is simplified to a binary classification problem:

- `0`: no heart disease
- `1`, `2`, `3`, or `4` in the original data: presence of heart disease
- After preparation, any original target value greater than zero is mapped to `1`.

The goal is to build an educational end-to-end ML system that can train a binary classifier and return a prediction with a positive-class probability through an HTTP API.

## Dataset Description

The project combines four commonly used UCI Heart Disease files from the [UCI Machine Learning Repository Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease):

| Source | File |
|---|---|
| Cleveland | `data/raw/processed.cleveland.data` |
| Hungarian | `data/raw/processed.hungarian.data` |
| Switzerland | `data/raw/processed.switzerland.data` |
| VA Long Beach | `data/raw/processed.va.data` |

The prepared dataset is `data/raw/heart.csv`. It has 920 source records and 14 columns: 13 predictors and the target. The predictor fields are `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, and `thal`. The target is binarized as described above.

The source files use `?` to represent missing measurements. The data-preparation notebook documents the combination, inspection, and target transformation. The training preprocessing code converts feature values to numeric, drops records with an unusable target and duplicate records, and imputes missing feature values within the fitted preprocessing pipeline.

The prepared CSV and UCI source files are present in this repository. To regenerate the combined file, open and run [`notebooks/data_preparation.ipynb`](notebooks/data_preparation.ipynb) from the repository root.

### Dataset acquisition

Download the official UCI archive from the [Heart Disease dataset page](https://archive.ics.uci.edu/dataset/45/heart+disease) or [direct dataset ZIP](https://archive.ics.uci.edu/static/public/45/heart+disease.zip). Extract these four files into `data/raw/`:

```text
processed.cleveland.data
processed.hungarian.data
processed.switzerland.data
processed.va.data
```

Run `notebooks/data_preparation.ipynb` to create `data/raw/heart.csv`. The source files and prepared CSV are also included in this repository.

## Installation

The project is tested in CI with Python 3.11. From the repository root, create and activate an isolated environment.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Project Structure

```text
.
├── app/
│   └── main.py                    # FastAPI inference service
├── data/
│   └── raw/
│       ├── processed.*.data       # UCI source datasets
│       └── heart.csv              # Combined, binary-target dataset
├── kubernetes/
│   ├── deployment.yaml            # Two-replica API Deployment
│   ├── monitoring.yaml             # Prometheus and Grafana monitoring stack
│   └── service.yaml               # API LoadBalancer Service
├── models/
│   ├── best_model.pkl             # Generated classifier (git-ignored)
│   ├── preprocessor.pkl           # Generated fitted transformer (git-ignored)
│   └── model_metrics.csv          # Model comparison output
├── notebooks/
│   ├── data_preparation.ipynb     # Dataset consolidation and validation
│   ├── eda.ipynb                  # Exploratory data analysis
│   └── mlops_architecture.ipynb   # Architecture and workflow description
├── src/
│   ├── preprocess.py              # Cleaning, splitting, and transformations
│   ├── train.py                   # Model training and evaluation
│   └── train_mlflow.py             # MLflow-tracked model training
├── tests/                          # Pytest suite
├── .github/workflows/ci.yml        # GitHub Actions workflow
├── .dockerignore
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

## Data Preprocessing

The preprocessing workflow in `src/preprocess.py`:

- Loads `data/raw/heart.csv` and verifies the target column exists.
- Converts data columns to numeric values and treats invalid values as missing.
- Removes rows with missing targets and duplicate records.
- Separates the binary target from the 13 predictors.
- Creates a stratified 80/20 train/test split with random seed `42`.
- Fits median imputation and standard scaling for numeric predictors.
- Fits most-frequent imputation and one-hot encoding for categorical predictors.
- Fits transformations using only the training data, then applies them to both partitions to avoid test-set leakage.
- Saves the fitted transformer to `models/preprocessor.pkl`.

Run preprocessing directly:

```bash
python src/preprocess.py
```

## Model Training

`src/train.py` trains and compares:

- **Logistic Regression** (`random_state=42`, `max_iter=1000`)
- **Random Forest**, tuning `n_estimators`, `max_depth`, and `min_samples_leaf`

Each model uses five-fold stratified cross-validation and grid search. Logistic Regression evaluates `C=[0.1, 1.0, 10.0]`; Random Forest evaluates `n_estimators=[100, 200]`, `max_depth=[None, 10]`, and `min_samples_leaf=[1, 2]`. Preprocessing is part of each fold's pipeline, preventing imputation, scaling, or encoding from using validation-fold data. The model with the highest mean cross-validation ROC-AUC is selected, with mean F1 as the tie-breaker. The held-out test set is kept separate for final evaluation. Results include the cross-validation mean and standard deviation and held-out accuracy, precision, recall, F1 score, and ROC-AUC. The selected classifier and its fitted preprocessor, comparison metrics, and confusion-matrix visualizations are saved under `models/` and `screenshots/`.

Train the models and generate the local model artifacts:

```bash
python src/train.py
```

The preprocessing and training commands expect the prepared dataset at `data/raw/heart.csv`. The serialized `.pkl` artifacts are generated locally and are excluded from Git.

## MLflow Tracking

`src/train_mlflow.py` performs the candidate-model training with MLflow. It logs each model as a separate run, including best search parameters, cross-validation and held-out evaluation metrics, the fitted preprocessing/model pipeline, confusion-matrix image, and model-comparison output. It selects the model by mean cross-validation ROC-AUC, using mean F1 as the tie-breaker, and writes the selected classifier and preprocessor locally.

Run the tracked training workflow:

```bash
python src/train_mlflow.py
```

By default, MLflow uses its local tracking store. Configure `MLFLOW_TRACKING_URI` to use another tracking server:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python src/train_mlflow.py
```

Start the local MLflow UI in another terminal:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Then visit <http://127.0.0.1:5000>. Local MLflow state is for development; configure a durable, access-controlled tracking backend for shared or production deployments. Do not commit credentials.

## FastAPI API

The application in `app/main.py` loads `models/best_model.pkl` and `models/preprocessor.pkl` during startup. It validates prediction input with Pydantic, transforms input using the stored preprocessor, and returns a binary class, human-readable label, and probability for class `1`.

Start the API after generating the model artifacts:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Available endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API information |
| `GET` | `/health` | Health status; returns healthy when model artifacts are loaded |
| `POST` | `/predict` | Predict heart disease presence for one record |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/metrics` | Prometheus-format HTTP instrumentation |

The `/metrics` route is intentionally excluded from the OpenAPI schema. API request middleware also logs method, route, status, duration, and client address.

### Example API requests

Check service health:

```bash
curl http://127.0.0.1:8000/health
```

Submit a prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63.0,
    "sex": 1,
    "cp": 1,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150.0,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0.0,
    "thal": 6.0
  }'
```

Example response shape (the prediction and probability depend on the trained artifacts):

```json
{
  "prediction": 1,
  "prediction_label": "Heart Disease",
  "probability": 0.73
}
```

Inspect exposed metrics:

```bash
curl http://127.0.0.1:8000/metrics
```

## Docker

Build the model artifacts before building the image. The Dockerfile copies the application project into a Python 3.11 slim image, installs dependencies, and runs Uvicorn as an unprivileged user.

```bash
python src/train.py
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
```

Visit <http://localhost:8000/docs> for Swagger UI, or call the API at `http://localhost:8000`. To stop the foreground container, press `Ctrl+C`; `--rm` removes the container after it stops.

## Kubernetes

The manifests in `kubernetes/` create a two-replica Deployment and a `LoadBalancer` Service. The Deployment defines CPU/memory requests and limits, restrictive container security settings, and startup, readiness, and liveness probes against `/health`. The Service maps port `80` to the API container port `8000`. `monitoring.yaml` deploys Prometheus with a scrape target for the API Service and Grafana with a provisioned Prometheus datasource and API dashboard.

Build the image first. For Docker Desktop Kubernetes, make the image available to its cluster, then apply the manifests:

```bash
docker build -t heart-disease-api:latest .
```

Create the Grafana password Secret before applying the monitoring manifest. In PowerShell:

```powershell
$env:GRAFANA_ADMIN_PASSWORD = Read-Host "Grafana admin password"
kubectl create secret generic grafana-admin --from-literal="admin-password=$env:GRAFANA_ADMIN_PASSWORD"
Remove-Item Env:GRAFANA_ADMIN_PASSWORD
```

Apply the Kubernetes manifests:

```bash
kubectl apply -f kubernetes/
kubectl rollout status deployment/heart-disease-api
```

Inspect pods, services, and application logs:

```bash
kubectl get pods -l app=heart-disease-api
kubectl get service heart-disease-api
kubectl logs deployment/heart-disease-api --all-containers=true --tail=100
```

Check the API Service's external address:

```bash
kubectl get service heart-disease-api
```

Use the assigned external address for `/docs`, `/health`, and `/metrics`. This requires the cluster to provide a load-balancer implementation. If `EXTERNAL-IP` remains pending in a local cluster, use port forwarding:

```bash
kubectl port-forward service/heart-disease-api 8001:80
```

Then use `http://127.0.0.1:8001/docs`, `http://127.0.0.1:8001/health`, or `http://127.0.0.1:8001/metrics`.

For a shared cluster, push a versioned or digest-pinned image to a registry accessible by the cluster and update the image reference in `kubernetes/deployment.yaml`. Registry authentication and deployment credentials should be configured securely, not committed to the repository.

### Prometheus and Grafana in Kubernetes

The Kubernetes manifests configure Prometheus to scrape the API Service every 15 seconds and provision Grafana with a Prometheus datasource and a basic API dashboard:

```bash
kubectl rollout status deployment/prometheus
kubectl port-forward service/prometheus 9090:9090
```

Open <http://127.0.0.1:9090/targets> and confirm the `heart-disease-api` target is **UP**. To open the Grafana dashboard:

```bash
kubectl rollout status deployment/grafana
kubectl port-forward service/grafana 3000:3000
```

Visit <http://127.0.0.1:3000>, sign in with username `admin` and the password used to create the Secret, and open the **Heart Disease API** dashboard. Prometheus and Grafana use ephemeral storage in this assignment setup; metrics and dashboard state are not durable across pod replacement.

## Monitoring

The FastAPI application logs HTTP method, path, status, duration, and client address. It is also instrumented with `prometheus-fastapi-instrumentator` and exposes request metrics at `/metrics`. The endpoint can be checked locally with:

```bash
curl http://127.0.0.1:8000/metrics
```

For Kubernetes, apply the manifests and port-forward the Prometheus Service to inspect its **Targets** page or the Grafana Service to view the provisioned dashboard. Persistent metrics storage and alert rules are not configured.

MLflow serves a different purpose: it tracks model-development experiments and evaluation artifacts, while Prometheus-compatible metrics describe API runtime behavior.

## Testing

Install project dependencies, then run the complete test suite and lint checks from the repository root:

```bash
python -m pytest -v
python -m flake8 --show-source --statistics src app tests
```

Tests cover preprocessing behavior, model artifact loading and predictions, and API routes including health, prediction, and metrics. The test fixture creates temporary model artifacts when trained `.pkl` files are absent; API startup and Docker image creation still require the real artifacts generated by training.

## CI/CD

The workflow at `.github/workflows/ci.yml` runs on pushes and pull requests targeting `main`. It:

1. Checks out the revision and sets up Python 3.11.
2. Installs dependencies from `requirements.txt`.
3. Runs Flake8 and pytest.
4. Checks for required source files and the Dockerfile.
5. Trains the model and verifies generated artifacts.
6. Builds the Docker image with plain build output.
7. Runs the container and verifies `/health` plus a sample `/predict` request.
8. Uploads model and screenshot artifacts for 14 days.

Any failed command fails the job. The workflow **builds** `heart-disease-api:ci` but does not push it to a registry or deploy it to Kubernetes. Automated publishing and deployment require a registry, appropriately scoped credentials, and deployment configuration.

## Results

The current saved model comparison (`models/model_metrics.csv`) reports five-fold stratified cross-validation and held-out test results:

| Model | CV ROC-AUC (mean ± SD) | CV F1 (mean ± SD) | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8789 ± 0.0439 | 0.8304 ± 0.0251 | 0.8315 | 0.8257 | 0.8824 | 0.8531 | **0.9213** |
| Random Forest | **0.8809 ± 0.0481** | **0.8442 ± 0.0239** | **0.8696** | **0.8545** | **0.9216** | **0.8868** | 0.9179 |

Model selection uses mean cross-validation ROC-AUC, with mean F1 as the tie-breaker; the current search selects Random Forest. The held-out test results are reported separately and are not used to choose the model. These figures are preliminary and not a clinical validation; further external validation and subgroup analysis are necessary before any real-world use.

## Future Improvements

- Add a reproducible data-versioning strategy and document dataset provenance and license terms.
- Pin and lock Python dependencies to improve build reproducibility.
- Extend hyperparameter search, calibration, threshold analysis, and external validation.
- Track dataset and model versions alongside MLflow experiments.
- Build and publish versioned, immutable Docker images to a container registry.
- Extend GitHub Actions with secure, approval-gated Kubernetes deployment.
- Add persistent Prometheus/Grafana storage and alert rules.
- Add API integration/load tests, vulnerability scanning, and container-image checks.
- Evaluate model fairness, drift monitoring, and retraining criteria.
- Improve application configuration, secrets management, and production logging.
