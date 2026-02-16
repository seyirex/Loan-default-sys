# Kubernetes Deployment — Design Choices

## Overview

Production-ready Kubernetes manifests for the loan default prediction system. Designed for reliability, autoscaling, and zero-downtime deployments without over-engineering for the current scale.

## Architecture

```
Ingress (nginx, TLS)
  ├── loan-api.yourdomain.com → loan-api Service (LoadBalancer)
  │                                └── loan-api Deployment (3 replicas)
  │                                      ├── /healthz  (liveness)
  │                                      └── /readyz   (readiness)
  └── mlflow.yourdomain.com  → mlflow-service (ClusterIP)
                                 └── mlflow-server Deployment (1 replica)

Internal Services (ClusterIP):
  ├── redis-service     → redis Deployment (1 replica)
  ├── mlflow-service    → mlflow-server Deployment (1 replica)
  └── postgres-service  → postgres Deployment (1 replica, optional)

Background Workers:
  └── celery-worker Deployment (2 replicas, no service — pulls from Redis)
```

## Design Choices

### Resource Requests & Limits

| Component | Request (CPU/Mem) | Limit (CPU/Mem) | Rationale |
|-----------|------------------|-----------------|-----------|
| **API** | 250m / 512Mi | 500m / 1Gi | Model inference is memory-bound (XGBoost model + scaler in memory). CPU is low because inference is fast (~10ms) |
| **Celery Worker** | 500m / 768Mi | 1000m / 2Gi | Higher limits for batch processing. Workers handle concurrent predictions and need headroom for SMOTE-sized payloads |
| **Redis** | 100m / 128Mi | 200m / 256Mi | Lightweight broker. Only stores job metadata and results, not large datasets |
| **MLflow** | 250m / 512Mi | 500m / 1Gi | Serves UI and artifact API. Single replica sufficient — not on the inference hot path |
| **Postgres** | 250m / 256Mi | 500m / 512Mi | Optional MLflow backend. Small footprint — only stores experiment metadata |

Requests are set to ~50% of limits to allow burstable QoS while guaranteeing baseline resources during scheduling.

### Health Probes

**API probes are split by purpose:**
- **Liveness** (`/healthz`): "Is the process alive?" Returns 200 always. Fails → K8s restarts the pod. Generous 30s initial delay and 10s timeout to avoid false restarts during model loading
- **Readiness** (`/readyz`): "Can it serve traffic?" Returns 503 if model not loaded. Fails → pod removed from Service endpoints (no traffic). 20s initial delay because model loading takes ~15-30s

This separation means a pod with a failed model load stays alive (debuggable) but receives no traffic. K8s automatically routes requests to healthy pods only.

**Redis and Postgres** use exec probes (`redis-cli ping`, `pg_isready`) — faster and more reliable than HTTP for data stores.

### Autoscaling (HPA)

- **Range:** 3-10 API pods. Minimum 3 for high availability across failure domains
- **Triggers:** CPU > 70% or Memory > 80%. CPU is the primary signal for inference workloads
- **Scale-up:** Aggressive — up to 100% increase or +2 pods per 30s, no stabilization delay. Prioritizes responsiveness to traffic spikes
- **Scale-down:** Conservative — max 50% decrease per 60s, 300s stabilization window. Prevents flapping during variable traffic

Celery workers are not autoscaled — batch jobs are not latency-sensitive and 2 replicas handle the expected load. Can add HPA on Redis queue depth if batch volume grows.

### Networking

- **API Service:** LoadBalancer type for direct external access. Switch to ClusterIP when Ingress is configured
- **Internal services** (Redis, MLflow, Postgres): ClusterIP only — no external exposure needed
- **Ingress:** nginx controller with TLS, rate limiting (100 RPS), CORS, and 300s proxy timeout for long batch requests

### Storage

- **mlflow-storage PVC (10Gi, ReadWriteMany):** Shared across API and Celery pods so both can access model artifacts. ReadWriteMany requires a CSI driver that supports it (EFS, NFS, etc.)
- **postgres-storage PVC (5Gi, ReadWriteOnce):** Exclusive to Postgres. ReadWriteOnce is fine for single-replica database

### Secrets

Current: base64-encoded Secret (template only, not production-safe). Production: use External Secrets Operator to sync from AWS Secrets Manager / GCP Secret Manager, or sealed-secrets for GitOps workflows.

## Deployment Order

```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml -f secret.yaml
kubectl apply -f pvc.yaml
kubectl apply -f service-redis.yaml -f service-mlflow.yaml -f service-api.yaml
kubectl apply -f deployment-redis.yaml
kubectl apply -f deployment-mlflow.yaml
kubectl apply -f deployment-api.yaml
kubectl apply -f deployment-celery-worker.yaml
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml
```

Redis and MLflow must be running before the API starts (API connects to both at startup).

## GPU Support (Not Implemented)

To enable GPU inference:
1. **Node pool:** Add GPU node pool with `nvidia.com/gpu` resource (e.g., `g4dn.xlarge` on AWS)
2. **Device plugin:** Install NVIDIA device plugin DaemonSet (`k8s-device-plugin`)
3. **Manifest changes:** Add `nvidia.com/gpu: 1` to API deployment resource limits and add `nodeSelector` or `tolerations` for GPU nodes
4. **Model:** Convert XGBoost to GPU-enabled predictor (`tree_method: 'gpu_hist'`) or switch to ONNX Runtime with CUDA

Not implemented because XGBoost CPU inference is already ~10ms — GPU adds infrastructure cost without meaningful latency improvement at current scale.

## How the Service Would Scale

- **Vertical:** Increase API resource limits and Uvicorn workers (4 → 8) per pod
- **Horizontal:** HPA already configured (3-10 pods). Increase max to 20+ for higher traffic
- **Infrastructure:** Cluster Autoscaler adds nodes when pods are unschedulable. Redis Cluster (3-6 nodes) for HA
- **Multi-model:** Separate Deployment per model type with Ingress path routing, enabling independent scaling and rollback per model
