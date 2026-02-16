# Loan Default Prediction - MLOps System

A production-ready ML inference system for loan default prediction demonstrating MLOps best practices.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.129.0-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-blue.svg)](https://kubernetes.io/)

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Workflow](#development-workflow)
- [API Documentation](#api-documentation)
- [API Examples](#api-examples)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This system provides a complete MLOps pipeline for loan default prediction, including:

- **Training Pipeline**: Production training script with MLflow tracking and model registry
- **Inference API**: FastAPI REST API with authentication and rate limiting
- **Batch Processing**: Async batch predictions using Celery
- **Drift Detection**: PSI-based feature drift monitoring
- **Metrics**: Prometheus metrics for observability
- **Containerization**: Docker and Kubernetes ready

**Model Performance:**
- Accuracy: 89%
- Recall: 77%
- F1 Score: 0.31
- ROC-AUC: 0.87

## 🏗️ Architecture

![Architecture Diagram](images/diagram.png)

### Request Flow Diagram

**Real-time Prediction Request (Docker Compose):**
```
┌──────────┐
│  Client  │
└────┬─────┘
     │ POST http://localhost:8005/api/v1/predict
     │ Header: X-API-Key: your-secret-api-key
     │ Body: {employed: 1, bank_balance: 10000, annual_salary: 50000}
     ▼
┌────────────────────────────────────────────────────┐
│         FastAPI Container (port 8005)              │
│  ┌──────────────────────────────────────────────┐ │
│  │ 1. Authentication Middleware                 │ │
│  │    • Verify X-API-Key header                 │ │
│  │    • Return 403 if invalid                   │ │
│  └──────────────┬───────────────────────────────┘ │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 2. Rate Limiting (SlowAPI)                   │ │
│  │    • Check: 100 requests/minute per IP       │ │
│  │    • Return 429 if limit exceeded            │ │
│  └──────────────┬───────────────────────────────┘ │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 3. Request Validation (Pydantic)             │ │
│  │    • Type checking (int, float)              │ │
│  │    • Range validation (salary > 0)           │ │
│  │    • Return 422 if invalid                   │ │
│  └──────────────┬───────────────────────────────┘ │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 4. Feature Engineering (preprocessing.py)    │ │
│  │    • Calculate Saving_Rate                   │ │
│  │    • Formula: Bank_Balance / Annual_Salary   │ │
│  │    • Create feature vector [4 features]      │ │
│  └──────────────┬───────────────────────────────┘ │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 5. Model Inference (ModelService)            │ │
│  │    • Retrieve cached model from app_state    │ │
│  │    • Apply StandardScaler (fit during train) │ │
│  │    • XGBoost.predict_proba() [~10ms]         │ │
│  │    • Get class (0/1) + probability           │ │
│  └──────────────┬───────────────────────────────┘ │
│                 │                                  │
│                 │ Model loaded from MLflow         │
│                 │ (once at startup)                │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │    MLflow Model Registry Access              │ │
│  │    • Model: file:///app/mlflow               │ │
│  │    • Stage: Production                       │ │
│  │    • Artifacts: model.pkl + scaler.pkl       │ │
│  └──────────────┬───────────────────────────────┘ │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 6. Drift Detection (DriftDetector)           │ │
│  │    • Sample: 10% of predictions              │ │
│  │    • Calculate PSI vs reference (1000 pred)  │ │
│  │    • Alert if PSI > 0.15 for any feature     │ │
│  │    • Async/non-blocking                      │ │
│  └──────────────┬───────────────────────────────┘ │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 7. Metrics Recording (Prometheus)            │ │
│  │    • loan_predictions_total++                │ │
│  │    • loan_prediction_duration_seconds        │ │
│  │    • loan_prediction_result_total{class=0}   │ │
│  │    • loan_model_drift_psi{feature=...}       │ │
│  └──────────────┬───────────────────────────────┘ │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 8. Response Formatting                       │ │
│  │    • Map: 0 → "no_default", 1 → "default"    │ │
│  │    • Risk: Low (<0.3), Medium, High (>0.7)   │ │
│  │    • Return JSON with model metadata         │ │
│  └──────────────┬───────────────────────────────┘ │
└─────────────────┼────────────────────────────────┘
                  ▼
     ┌────────────────────────────────┐
     │  Client receives response      │
     │  Status: 200 OK                │
     │  {                             │
     │    "success": true,            │
     │    "data": {                   │
     │      "prediction": 0,          │
     │      "probability": 0.0823,    │
     │      "default_risk": "Low",    │
     │      "model_version": "1"      │
     │    }                           │
     │  }                             │
     │  Total Duration: ~15-25ms      │
     └────────────────────────────────┘
```

**Batch Prediction Request (Celery + Redis):**
```
┌──────────┐
│  Client  │
└────┬─────┘
     │ POST http://localhost:8005/api/v1/predict/batch
     │ Body: {predictions: [{...}, {...}, ...]}  (up to 1000)
     ▼
┌────────────────────────────────────────────────────┐
│         FastAPI Container (port 8005)              │
│  1. Validate request (max 1000 predictions)        │
│  2. Create Celery task via batch_service.py        │
│  3. Return job_id immediately (202 Accepted)       │
│     Duration: ~5ms                                 │
└────┬───────────────────────────────────────────────┘
     │
     │ Task queued via Celery
     ▼
┌────────────────────────────────────────────────────┐
│      Redis Container (port 6389)                   │
│  • Broker: Stores task in "celery" queue           │
│  • Task data: job_id, predictions array, metadata  │
└────┬───────────────────────────────────────────────┘
     │
     │ Celery worker polls queue every 1s
     ▼
┌────────────────────────────────────────────────────┐
│    Celery Worker Container (2 concurrent workers)  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 1. Receive task from Redis broker            │ │
│  └──────────────┬───────────────────────────────┘ │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 2. Load model from MLflow                    │ │
│  │    • Same model as API (shared volume)       │ │
│  │    • Cached after first load                 │ │
│  └──────────────┬───────────────────────────────┘ │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 3. Batch Processing                          │ │
│  │    • Process in chunks of 100                │ │
│  │    • Feature engineering for each row        │ │
│  │    • Model inference (vectorized)            │ │
│  │    • ~100ms per 100 predictions              │ │
│  └──────────────┬───────────────────────────────┘ │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 4. Drift Detection (5% sample rate)          │ │
│  │    • Lower sampling for batch vs online      │ │
│  │    • Same PSI calculation                    │ │
│  └──────────────┬───────────────────────────────┘ │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 5. Store results in Redis                    │ │
│  │    • Key: celery-task-meta-{job_id}          │ │
│  │    • Value: {status, result, traceback}      │ │
│  │    • TTL: 24 hours                           │ │
│  └──────────────┬───────────────────────────────┘ │
│                 ▼                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ 6. Task Complete                             │ │
│  │    • Update status: PENDING → SUCCESS        │ │
│  │    • Log completion with duration            │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
     │
     │ Results stored in Redis
     ▼
┌────────────────────────────────────────────────────┐
│      Redis Container (Result Backend)              │
│  • Results accessible via job_id                   │
│  • Client polls: GET /api/v1/predict/batch/{id}    │
└─────────────────────────────────────────────────┬──┘
                                                   │
              Client polling every 2-5s             │
                                                   ▼
                    ┌───────────────────────────────────┐
                    │  FastAPI returns result           │
                    │  {                                │
                    │    "job_id": "abc-123",           │
                    │    "status": "SUCCESS",           │
                    │    "total_predictions": 500,      │
                    │    "results": [                   │
                    │      {"prediction": 0, "prob": 0.1},│
                    │      ...                          │
                    │    ],                             │
                    │    "processing_time": "2.3s"      │
                    │  }                                │
                    └───────────────────────────────────┘
```

**Model Loading at Startup:**
```
Docker Container Start
         │
         ▼
┌─────────────────────────────────────┐
│  FastAPI lifespan event (main.py)  │
│  Triggered once at startup          │
└────┬────────────────────────────────┘
     ▼
┌─────────────────────────────────────┐
│  ModelService.load_model()          │
│  1. Connect to MLflow tracking URI  │
│     • file:///app/mlflow (local)    │
│  2. Query registry for model        │
│     • Name: "loan-default-xgboost"  │
│     • Stage: "Production"           │
│  3. Download artifacts               │
│     • model.pkl                     │
│     • scaler.pkl                    │
│  4. Load into memory (~30s)         │
│  5. Store in app_state              │
└────┬────────────────────────────────┘
     ▼
┌─────────────────────────────────────┐
│  Readiness probe succeeds           │
│  GET /readyz returns 200            │
│  Container ready for traffic        │
└─────────────────────────────────────┘
```

**Monitoring & Observability:**
```
┌────────────────────────────────────────────────┐
│           Prometheus Scraper                   │
│  • Scrapes: http://localhost:8005/metrics      │
│  • Interval: 15s                               │
│  • Stores time-series data                     │
└────┬───────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────┐
│  Custom Metrics Exposed                        │
│  • loan_predictions_total (counter)            │
│  • loan_prediction_duration_seconds (histogram)│
│  • loan_prediction_result_total (counter)      │
│  • loan_model_drift_psi (gauge)                │
│  • loan_model_drift_detected (gauge)           │
└────────────────────────────────────────────────┘
```

**Key Flow Characteristics:**
- **Latency**: Real-time predictions: 15-25ms, Batch: ~10ms per prediction
- **Throughput**: 4 Uvicorn workers = ~400 req/sec per container
- **State**: Stateless API, state in MLflow (models) and Redis (jobs)
- **Scalability**: Horizontal scaling via Docker Compose replicas or K8s HPA
- **Resilience**: Health checks, retry logic (Celery), graceful degradation

## ✨ Features

### Core Functionality
- ✅ Real-time loan default prediction API
- ✅ Batch prediction processing with job tracking
- ✅ Automatic feature engineering (Saving Rate calculation)
- ✅ SMOTE-based class balancing
- ✅ Production model auto-promotion based on metrics

### MLOps Best Practices
- ✅ MLflow experiment tracking and model registry
- ✅ Population Stability Index (PSI) drift detection
- ✅ Prometheus metrics and monitoring
- ✅ API key authentication
- ✅ Rate limiting (100 req/min)
- ✅ Structured logging with Loguru
- ✅ Health checks for Kubernetes probes

### Infrastructure
- ✅ Docker containerization (Python 3.11-slim images ~300MB)
- ✅ Docker Compose orchestration (API, MLflow, Redis, Celery)
- ✅ Hot-reloading for local development
- ✅ Kubernetes manifests with HPA (3-10 pod autoscaling)
- ✅ Resource limits and requests
- ✅ Persistent volumes for MLflow artifacts and database
- ✅ Health checks and readiness probes

## 🔧 Prerequisites

- **Docker** (20.10+) and Docker Compose (2.0+)
- **Python** 3.11 (only if running outside Docker)
- **Kubernetes** cluster (optional, for K8s deployment)
- **kubectl** (optional, for K8s deployment)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
cd loan-default-sys

# Create environment file
cp .env.example .env

# Edit .env and set your API key
# Change: API_KEY=your-secret-api-key-here-change-in-production
```

### 2. Build Docker Images

```bash
docker-compose build
```

### 3. Train the Model

```bash
docker-compose run --rm api python training/train.py
```

This will:
- Load and preprocess data
- Train XGBoost with SMOTE
- Log metrics to MLflow
- Save model to registry
- Auto-promote to Production stage if metrics meet criteria

**View training results:**
- Open MLflow UI at http://localhost:5001
- Navigate to "loan-default-prediction" experiment
- View metrics, parameters, and model artifacts

**Transfer trained model to another server:**

The trained model and MLflow artifacts are stored in `./mlflow` directory. To deploy on another server without retraining:

```bash
# On training server - package MLflow artifacts
tar -czf mlflow-artifacts.tar.gz mlflow/

# Transfer to deployment server
scp mlflow-artifacts.tar.gz user@deployment-server:/path/to/loan-default-sys/

# On deployment server - extract artifacts
tar -xzf mlflow-artifacts.tar.gz

# Start services (skip training step)
docker-compose up
```

The API will automatically load the Production model from the `./mlflow` directory at startup.

### 4. Start Services

```bash
docker-compose up
```

Services will be available at:
- **API**: http://localhost:8005
- **API Docs**: http://localhost:8005/docs
- **Health Check**: http://localhost:8005/healthz
- **Metrics**: http://localhost:8005/metrics
- **MLflow UI**: http://localhost:5001
- **Redis**: localhost:6389

### 5. Make a Prediction

```bash
curl -X POST "http://localhost:8005/api/v1/predict" \
  -H "X-API-Key: your-secret-api-key-here-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "employed": 1,
    "bank_balance": 10000.0,
    "annual_salary": 50000.0
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "prediction": 0,
    "probability": 0.0823,
    "default_risk": "Low",
    "model_version": "1",
    "features_used": {
      "employed": 1,
      "bank_balance": 10000.0,
      "annual_salary": 50000.0
    }
  },
  "message": "Prediction completed successfully"
}
```

**📚 For more examples, see [EXAMPLES.md](EXAMPLES.md)**

## 🔄 Development Workflow

### Docker Compose Services

The system runs 4 containerized services:

| Service | Port | Description |
|---------|------|-------------|
| **api** | 8005 | FastAPI inference service with auto-reload |
| **mlflow** | 5001 | MLflow tracking UI and model registry |
| **redis** | 6389 | Message broker for Celery tasks |
| **celery-worker** | - | Background worker for batch predictions |

```bash
# Start all services
docker-compose up

# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f api          # API logs only
docker-compose logs -f              # All services

# Stop services
docker-compose down

# Rebuild after dependency changes
docker-compose up --build
```

### Local Development with Hot-Reloading

The docker-compose setup includes hot-reloading for rapid development:

```bash
# Start services with auto-reload
docker-compose up

# Edit files in src/, tests/, or training/ - changes apply immediately
# API automatically reloads when you save Python files
```

**Mounted Volumes:**
- `./src` → `/app/src` - API code changes reload automatically
- `./tests` → `/app/tests` - Test changes reflect immediately
- `./training` → `/app/training` - Training script updates available instantly

### Running Tests

```bash
# Run all tests with coverage (HTML report auto-generated)
docker-compose run --rm api pytest

# Run specific test file
docker-compose run --rm api pytest tests/test_api/test_health.py -v

# Run tests by marker
docker-compose run --rm api pytest -m unit
docker-compose run --rm api pytest -m integration
docker-compose run --rm api pytest -m slow

# View HTML coverage report (opens in browser)
open htmlcov/index.html
```

### Code Quality

```bash
# Format code with Black (88 character line length)
docker-compose run --rm api black src/ tests/

# Lint with Ruff (auto-fix issues)
docker-compose run --rm api ruff check src/ tests/ --fix

# Type check with mypy
docker-compose run --rm api mypy src/

# Run all quality checks
docker-compose run --rm api black src/ tests/ && \
docker-compose run --rm api ruff check src/ tests/ --fix && \
docker-compose run --rm api mypy src/
```

### View Logs

```bash
# API logs
docker-compose logs -f api

# Celery worker logs
docker-compose logs -f celery-worker

# All logs
docker-compose logs -f
```

## 📚 API Documentation

### Authentication

All protected endpoints require an API key in the header:

```bash
X-API-Key: your-secret-api-key
```

### Endpoints

#### `GET /healthz` - Liveness Probe
Returns 200 if service is alive.

#### `GET /readyz` - Readiness Probe
Returns 200 if service is ready (model loaded).

#### `POST /api/v1/predict` - Real-time Prediction
Make a single prediction.

**Request Body:**
```json
{
  "employed": 1,
  "bank_balance": 10000.0,
  "annual_salary": 50000.0
}
```

**Rate Limit:** 100 requests/minute

#### `POST /api/v1/predict/batch` - Submit Batch Job
Submit batch predictions for async processing.

**Request Body:**
```json
{
  "predictions": [
    {"employed": 1, "bank_balance": 10000, "annual_salary": 50000},
    {"employed": 0, "bank_balance": 5000, "annual_salary": 30000}
  ]
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "PENDING",
  "message": "Batch job submitted successfully",
  "total_predictions": 2
}
```

#### `GET /api/v1/predict/batch/{job_id}` - Check Batch Status
Check status of batch job.

#### `GET /api/v1/model/info` - Model Information
Get information about deployed model.

#### `GET /metrics` - Prometheus Metrics
Prometheus-formatted metrics (no auth required).

---

## 📚 API Examples

For comprehensive examples including curl commands and full request/response pairs, see **[EXAMPLES.md](EXAMPLES.md)**.

The examples file includes:
- ✅ Single predictions (low, medium, high risk scenarios)
- ✅ Batch prediction workflow
- ✅ Model information queries
- ✅ Health check examples
- ✅ Error cases and validation examples
- ✅ Input/output field specifications

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (Minikube, GKE, EKS, AKS, etc.)
- kubectl configured
- Container image pushed to registry

### Build and Push Image

```bash
# Build image
docker build -t your-registry/loan-default-api:latest .

# Push to registry
docker push your-registry/loan-default-api:latest

# Update k8s/deployment-api.yaml and k8s/deployment-celery-worker.yaml
# Change: image: loan-default-api:latest
# To: image: your-registry/loan-default-api:latest
```

### Deploy to Kubernetes

```bash
# Create namespace and resources
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/pvc.yaml

# Deploy services
kubectl apply -f k8s/deployment-redis.yaml
kubectl apply -f k8s/service-redis.yaml
kubectl apply -f k8s/deployment-api.yaml
kubectl apply -f k8s/service-api.yaml
kubectl apply -f k8s/deployment-celery-worker.yaml
kubectl apply -f k8s/hpa.yaml
```

### Verify Deployment

```bash
# Check pods
kubectl get pods -n loan-default-prediction

# Check services
kubectl get svc -n loan-default-prediction

# Check HPA
kubectl get hpa -n loan-default-prediction

# View logs
kubectl logs -f deployment/loan-api -n loan-default-prediction
```

### Access API

```bash
# Port forward to access API
kubectl port-forward svc/loan-api 8005:80 -n loan-default-prediction

# API now available at http://localhost:8005
```

## 📊 Monitoring

### Prometheus Metrics

The `/metrics` endpoint exposes:

**Custom Metrics:**
- `loan_predictions_total` - Total predictions counter
- `loan_prediction_duration_seconds` - Prediction latency histogram
- `loan_prediction_result_total` - Prediction results (default/no_default)
- `loan_model_drift_psi` - PSI score per feature
- `loan_model_drift_detected` - Drift detection binary indicator

**Standard Metrics:**
- HTTP request counts, latencies, and status codes
- Process metrics (CPU, memory, etc.)

### Drift Detection

The system automatically monitors feature drift using PSI:

- **Reference Data**: First 1000 predictions
- **Window Size**: 100 recent predictions
- **Threshold**: PSI > 0.15 triggers alert

Drift alerts are logged and exposed via Prometheus metrics.

## 🧪 Testing

Test coverage: **>70%**

The project uses pytest with configured markers and automatic coverage reporting.

```bash
# Run all tests (generates both terminal and HTML coverage reports)
docker-compose run --rm api pytest

# Run with specific markers
docker-compose run --rm api pytest -m unit         # Unit tests only
docker-compose run --rm api pytest -m integration  # Integration tests only
docker-compose run --rm api pytest -m slow         # Slow running tests only

# Run specific test directories
docker-compose run --rm api pytest tests/test_api/ -v
docker-compose run --rm api pytest tests/test_services/ -v

# View detailed coverage report
open htmlcov/index.html
```

**Test Structure:**
```
tests/
├── conftest.py              # Shared fixtures
├── test_api/
│   ├── test_health.py       # Health endpoint tests
│   ├── test_predict.py      # Prediction endpoint tests
│   └── test_model.py        # Model endpoint tests
└── test_services/
    ├── test_model_service.py      # Model loading tests
    ├── test_batch_service.py      # Celery batch tests
    ├── test_drift_detector.py     # Drift detection tests
    ├── test_metrics_service.py    # Prometheus metrics tests
    └── test_training_service.py   # Training pipeline tests
```

## 📁 Project Structure

```
loan-default-sys/
├── src/
│   ├── api/v1/
│   │   ├── health.py         # Health check endpoints
│   │   ├── predict.py        # Prediction endpoints
│   │   └── model.py          # Model info endpoint
│   ├── services/
│   │   ├── model_service.py    # Model loading and inference
│   │   ├── training_service.py # Model training orchestration
│   │   ├── drift_detector.py   # PSI drift detection
│   │   ├── metrics_service.py  # Prometheus metrics
│   │   └── batch_service.py    # Celery tasks
│   ├── schemas/
│   │   ├── health.py         # Health schemas
│   │   └── prediction.py     # Prediction schemas
│   ├── utils/
│   │   └── preprocessing.py  # Utility functions
│   ├── config.py             # Configuration
│   ├── logging_config.py     # Logging setup
│   └── main.py               # FastAPI app
├── training/
│   ├── train.py              # Training script
│   ├── Default_Fin.csv       # Dataset
│   └── loan-default-prediction.ipynb
├── tests/
│   ├── test_api/             # API tests
│   └── test_services/        # Service tests
├── k8s/                      # Kubernetes manifests
├── mlflow/                   # MLflow artifacts
├── Dockerfile                # Container image
├── docker-compose.yml        # Local development
├── requirements.txt          # Python dependencies
├── pytest.ini                # Pytest configuration
├── .env.example              # Environment template
├── README.md                 # This file
├── EXAMPLES.md               # API request/response examples
└── DESIGN.md                 # Design document
```

## 🔐 Environment Variables

See [.env.example](.env.example) for all configuration options.

**Key Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | API authentication key | `your-secret-api-key` |
| `MODEL_STAGE` | MLflow model stage to load | `Production` |
| `RATE_LIMIT_PER_MINUTE` | Rate limit for `/predict` | `100` |
| `DRIFT_PSI_THRESHOLD` | PSI threshold for drift alert | `0.15` |
| `DRIFT_SAMPLING_RATE` | Probability of drift check per request | `0.1` (10%) |
| `LOG_LEVEL` | Logging level | `INFO` |
| `REDIS_HOST` | Redis hostname | `redis` |
| `REDIS_PORT` | Redis port | `6379` |

**Optional MLflow Auth (set in docker-compose.yml):**
- `MLFLOW_TRACKING_USERNAME` - MLflow UI username
- `MLFLOW_TRACKING_PASSWORD` - MLflow UI password
- `MLFLOW_FLASK_SERVER_SECRET_KEY` - Flask secret key for MLflow

## 🐛 Troubleshooting

### Model not loading

**Issue**: `Model service not initialized` error

**Solution**:
1. Ensure you've trained the model: `docker-compose run --rm api python training/train.py`
2. Check MLflow directory: `ls -la mlflow/`
3. Check logs: `docker-compose logs api`

### Permission denied errors (on server deployments)

**Issue**: `Permission denied: '/app/mlruns/...'` or `Failed to load model` with permission errors

**Solution**:
```bash
# Run the permission fix script
./fix-mlflow-permissions.sh

# Or manually fix permissions
sudo chown -R 1000:1000 mlflow/
sudo chmod -R 755 mlflow/

# Restart services
docker-compose restart
```

**Note**: This is common when transferring MLflow artifacts between servers or after copying from Docker volumes.

### Database locked errors

**Issue**: `sqlite3.OperationalError: database is locked` in MLflow logs

**Solution**:
SQLite doesn't handle concurrent access well. The docker-compose.yml is configured to enable WAL mode automatically, but if you're seeing this error:

```bash
# Manually enable WAL mode
sqlite3 mlflow/mlflow.db "PRAGMA journal_mode=WAL;"

# Or run the fix script which includes this
./fix-mlflow-permissions.sh

# Restart services
docker-compose restart
```

**Production recommendation**: For high-concurrency production deployments, migrate MLflow to PostgreSQL backend (see k8s/deployment-postgres.yaml).

### Celery worker not processing jobs

**Issue**: Batch jobs stuck in PENDING

**Solution**:
1. Check Redis is running: `docker-compose ps redis`
2. Verify Redis health: `redis-cli -h localhost -p 6389 ping` (should return PONG)
3. Check Celery logs: `docker-compose logs -f celery-worker`
4. Restart services: `docker-compose restart celery-worker redis`

### API key errors

**Issue**: 403 Forbidden errors

**Solution**:
1. Check .env file has correct API key
2. Ensure header is `X-API-Key` (case-sensitive)
3. Restart services after changing .env: `docker-compose restart api`

### Out of memory errors

**Issue**: Container crashes with OOM

**Solution**:
1. Increase Docker Desktop memory allocation (8GB+ recommended)
2. Reduce batch size in batch predictions
3. Adjust resource limits in k8s manifests

## 📄 License

This project is created for educational and demonstration purposes.

## 👥 Author

Created as an MLOps take-home exercise demonstrating production ML system design.

---

