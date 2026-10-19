# heart-disease-pipeline-mlops-bits-assignment
MLOPS Practice

## Heart Disease Prediction — MLOps Assignment

A BITS Pilani MLOps assignment scaffold for developing and deploying a heart
disease prediction service. The project is organized to support a reproducible
machine-learning workflow with **scikit-learn**, **MLflow**, **FastAPI**,
**Docker**, **GitHub Actions**, and **Kubernetes**.

> **Project status:** Data preparation, EDA, model preprocessing and training,
> MLflow tracking, the FastAPI service, pytest coverage, GitHub Actions CI, and
> baseline Kubernetes manifests are implemented. Review the deployment notes
> below and provide a registry image before deploying to a shared cluster.

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
6. Deploy the API using the manifests under `kubernetes/` after making the
   container image available to the target cluster.

## Repository layout

```text
data/raw/                 Raw input data (not committed)
notebooks/                Exploration and experimentation
src/                      Data and model pipeline code
app/                      FastAPI inference service
models/                   Generated model artifacts
tests/                    Automated tests
kubernetes/               Kubernetes Deployment and Service manifests
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

The Dockerfile packages the FastAPI service at `app.main:app`. Build and run
the container with:

```bash
docker build -t heart-disease-api .
docker run --rm -p 8000:8000 heart-disease-api
```

The API documentation is available at `http://localhost:8000/docs`, and its
health endpoint is `http://localhost:8000/health`.

## Kubernetes deployment

The manifests in `kubernetes/` run two API replicas behind an internal
`ClusterIP` service. They include resource requests and limits, restrictive
container security settings, and startup, readiness, and liveness probes
against the API's `/health` endpoint.

Build the image locally before applying the manifests:

```bash
docker build -t heart-disease-api:latest .
kubectl apply -f kubernetes/
```

The local image tag is suitable for Docker Desktop Kubernetes. For a shared
cluster, push the image to an accessible container registry, update the image
reference in `kubernetes/deployment.yaml` to a versioned or digest-pinned tag,
and configure registry credentials if needed before applying.

Inspect the deployment and application logs with:

```bash
kubectl get pods -l app=heart-disease-api
kubectl get services heart-disease-api
kubectl logs -l app=heart-disease-api --all-containers=true --tail=100
```

Access the API locally through port forwarding:

```bash
kubectl port-forward service/heart-disease-api 8000:80
```

Then open `http://localhost:8000/docs` for Swagger UI or
`http://localhost:8000/health` to check application health. Keep secrets out
of manifests and source control, use least-privilege cluster credentials, and
review resource requests and limits against measured workload behavior.

## CI/CD

The `.github/workflows/ci.yml` workflow installs dependencies, runs Flake8 and
pytest, verifies required files, trains the models, and uploads generated
artifacts on pushes and pull requests targeting `main`. Add registry
credentials and cluster access only as GitHub Actions secrets if automated
deployment is introduced.

## Testing

The test suite is under `tests/`. Run it with:

```bash
pytest
flake8 src app tests
```

## Assignment scope

This repository is intended for educational use as a BITS Pilani MLOps
assignment. Record dataset provenance, evaluation methodology, limitations,
and deployment assumptions in the project documentation as they are
implemented.
