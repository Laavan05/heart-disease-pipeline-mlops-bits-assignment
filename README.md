# heart-disease-pipeline-mlops-bits-assignment
MLOPS Practice

## Heart Disease Prediction — MLOps Assignment

A BITS Pilani MLOps assignment scaffold for developing and deploying a heart
disease prediction service. The project is organized to support a reproducible
machine-learning workflow with **scikit-learn**, **MLflow**, **FastAPI**,
**Docker**, **GitHub Actions**, and **Kubernetes**.

> **Project status:** This repository currently contains the initial project
> structure and environment configuration. The training pipeline, API
> implementation, GitHub Actions workflow, and Kubernetes manifests are not
> implemented yet.

## Intended workflow

1. Keep source data and data-preparation notebooks organized under `data/` and
   `notebooks/`; validate and document the dataset before training.
2. Build a scikit-learn training pipeline in `src/` and track parameters,
   metrics, and artifacts with MLflow.
3. Save deployable model artifacts under `models/` (generated model binaries
   are excluded from Git).
4. Serve model predictions through a FastAPI application in `app/`.
5. Package the service in a Docker image and use GitHub Actions for linting,
   tests, and image validation.
6. Add Kubernetes manifests under `deployment/` when the service is ready for
   cluster deployment.

## Repository layout

```text
data/raw/                 Raw input data (not committed)
notebooks/                Exploration and experimentation
src/                      Data and model pipeline code
app/                      FastAPI inference service
models/                   Generated model artifacts
tests/                    Automated tests
deployment/               Container and Kubernetes deployment resources
screenshots/              Assignment screenshots and evidence
.github/workflows/        GitHub Actions workflow definitions
Dockerfile                Container build scaffold
requirements.txt          Python dependencies
```

## Getting started

Use Python 3.10 or later. From the repository root, create and activate the
virtual environment on Windows:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux, activate it with:

```bash
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place any approved dataset in `data/raw/`. Do not commit private or
license-restricted data. No dataset is included in this scaffold.

## Experiment tracking

The dependency list includes MLflow. Once an experiment has been added, start
the local tracking UI with:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

The local `mlruns/` store is ignored by Git. Configure a remote tracking
backend through environment variables or deployment secrets when collaborating;
do not commit credentials.

## API and Docker

The Dockerfile is a starting point for the FastAPI service and expects the
application entry point `app.main:app` to be implemented. After adding that
entry point, build and run the container:

```bash
docker build -t heart-disease-api .
docker run --rm -p 8000:8000 heart-disease-api
```

Document the finalized request/response schema and health endpoint alongside
the API implementation before publishing the service.

## CI/CD and Kubernetes

The `.github/workflows/` directory is ready for workflows that install
dependencies, run `flake8` and `pytest`, and validate the Docker image on
pull requests. The `deployment/` directory is ready for Kubernetes resources
such as a Deployment and Service. Add registry credentials and cluster access
only as GitHub Actions secrets; no workflow or Kubernetes manifest is included
yet.

## Testing

Add tests under `tests/` as pipeline and API behavior is implemented, then run:

```bash
pytest
flake8 src app tests
```

## Assignment scope

This repository is intended for educational use as a BITS Pilani MLOps
assignment. Record dataset provenance, evaluation methodology, limitations,
and deployment assumptions in the project documentation as they are
implemented.
