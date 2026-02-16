# Kubernetes Deployment Guide - Loan Default Prediction System

## 📋 Overview

This directory contains Kubernetes manifests for deploying a production-ready ML prediction system with:
- **FastAPI API** (auto-scaling, 3-10 replicas)
- **MLflow Server** (experiment tracking & model serving with UI)
- **Redis** (message broker for Celery)
- **Celery Workers** (batch predictions)
- **Ingress** (smart routing with TLS support)
- **PostgreSQL** (optional, for MLflow backend)

---

## 🏗️ Architecture Diagram

```
                         Internet
                            |
                    ┌───────▼───────┐
                    │    Ingress    │
                    │ (nginx/ALB)   │
                    └───┬───────┬───┘
                        │       │
        ┌───────────────┘       └──────────────┐
        │                                      │
┌───────▼───────┐                    ┌────────▼────────┐
│  loan-api:80  │                    │ mlflow:5000     │
│  (FastAPI)    │                    │ (MLflow UI)     │
└───┬───────┬───┘                    └────────┬────────┘
    │       │                                 │
    │   ┌───▼────────────┐                    │
    │   │  Redis:6379    │                    │
    │   │  (Celery)      │                    │
    │   └───┬────────────┘                    │
    │       │                                 │
    │   ┌───▼────────────┐                    │
    │   │ Celery Workers │                    │
    │   │   (2 replicas) │                    │
    │   └───┬────────────┘                    │
    │       │                                 │
    └───────┴───────┬─────────────────────────┘
                    │
          ┌─────────▼─────────┐
          │  MLflow Storage   │
          │  (PVC - 10Gi)     │
          └───────────────────┘
```

---

## 📁 File Inventory

### Core Infrastructure (Required)
| File | Purpose | What It Does |
|------|---------|--------------|
| `namespace.yaml` | Isolated workspace | Creates the `loan-default-prediction` namespace |
| `configmap.yaml` | App configuration | Stores all non-sensitive settings (API ports, MLflow URI, thresholds) |
| `secret.yaml` | Credentials | Stores API keys (base64 encoded) |
| `pvc.yaml` | Persistent storage | 10Gi shared storage for MLflow models/artifacts |

### Application Components (Required)
| File | Purpose | Replicas | Resources |
|------|---------|----------|-----------|
| `deployment-redis.yaml` | Message broker | 1 | 256Mi RAM, 200m CPU |
| `service-redis.yaml` | Redis network access | - | ClusterIP (internal) |
| `deployment-api.yaml` | FastAPI servers | 3-10 (auto-scaled) | 1Gi RAM, 500m CPU |
| `service-api.yaml` | API load balancer | - | LoadBalancer (public) |
| `deployment-celery-worker.yaml` | Batch processing | 2 | 2Gi RAM, 1 CPU |

### MLflow Server (New! Recommended)
| File | Purpose | Why You Need It |
|------|---------|-----------------|
| `deployment-mlflow.yaml` | MLflow tracking server | Provides web UI for experiments, model registry |
| `service-mlflow.yaml` | MLflow network access | Exposes MLflow UI via Ingress |

### Optional Production Enhancements
| File | Purpose | When to Use |
|------|---------|-------------|
| `deployment-postgres.yaml` | PostgreSQL database | Production MLflow backend (instead of SQLite) |
| `service-postgres.yaml` | PostgreSQL access | Required if using PostgreSQL |
| `pvc-postgres.yaml` | PostgreSQL storage | Required if using PostgreSQL |

### Networking & Scaling
| File | Purpose | Features |
|------|---------|----------|
| `ingress.yaml` | Smart routing | Routes traffic to API and MLflow, TLS, rate limiting |
| `hpa.yaml` | Auto-scaling | Scales API 3→10 pods based on CPU/memory |

---

## 🚀 Deployment Options

### Option 1: Simple Setup (SQLite + File Storage)
**Best for:** Development, testing, small-scale production

**What you get:**
- ✅ MLflow UI for tracking experiments
- ✅ Shared file storage for models
- ⚠️ SQLite backend (single-user)

**Deploy:**
```bash
kubectl apply -f namespace.yaml
kubectl apply -f pvc.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f deployment-redis.yaml
kubectl apply -f service-redis.yaml
kubectl apply -f deployment-mlflow.yaml  # NEW!
kubectl apply -f service-mlflow.yaml     # NEW!
kubectl apply -f deployment-api.yaml
kubectl apply -f service-api.yaml
kubectl apply -f deployment-celery-worker.yaml
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml            # NEW!
```

### Option 2: Production Setup (PostgreSQL + S3/GCS)
**Best for:** Multi-user teams, high availability

**What you get:**
- ✅ PostgreSQL backend (multi-user, ACID compliant)
- ✅ External object storage (S3/GCS/Azure Blob)
- ✅ Better performance and reliability

**Additional steps:**
1. Deploy PostgreSQL:
   ```bash
   kubectl apply -f pvc-postgres.yaml
   kubectl apply -f deployment-postgres.yaml
   kubectl apply -f service-postgres.yaml
   ```

2. Update `deployment-mlflow.yaml`:
   ```yaml
   - --backend-store-uri
   - "postgresql://mlflow:mlflow@postgres-service:5432/mlflow"
   - --default-artifact-root
   - "s3://your-bucket/mlflow-artifacts"  # or gs:// for GCS
   ```

3. Add AWS/GCP credentials to the secret

---

## 🔧 Before Deployment Checklist

### 1. Build & Push Docker Image
```bash
# Build your image
docker build -t loan-default-api:latest .

# Tag for your registry
docker tag loan-default-api:latest your-registry/loan-default-api:latest

# Push to registry
docker push your-registry/loan-default-api:latest
```

### 2. Update Image References
Edit these files to use your registry:
- `deployment-api.yaml` line 20
- `deployment-celery-worker.yaml` line 20

```yaml
image: your-registry/loan-default-api:latest
```

### 3. Configure Secrets
**Update `secret.yaml`:**
```bash
# Generate base64 encoded secrets
echo -n "your-actual-api-key" | base64
echo -n "your-postgres-password" | base64

# Update secret.yaml with the output
```

**Never commit real secrets to git!** Use:
- [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets)
- [External Secrets Operator](https://external-secrets.io/)
- Cloud provider secret managers (AWS Secrets Manager, GCP Secret Manager)

### 4. Update Ingress Domains
Edit `ingress.yaml` lines 43, 53:
```yaml
- host: loan-api.yourdomain.com     # Your actual domain
- host: mlflow.yourdomain.com       # Your actual domain
```

### 5. Set Up TLS Certificates
```bash
# Option 1: cert-manager (auto-renewal)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Option 2: Manual certificate
kubectl create secret tls loan-prediction-tls \
  --cert=path/to/cert.crt \
  --key=path/to/cert.key \
  -n loan-default-prediction
```

---

## 📊 Deployment Instructions

### Step 1: Set Up Kubernetes Cluster
Choose your environment:

**Local (Minikube):**
```bash
minikube start --cpus=4 --memory=8192
minikube addons enable ingress
```

**AWS (EKS):**
```bash
eksctl create cluster \
  --name loan-prediction \
  --region us-west-2 \
  --nodes 3 \
  --node-type t3.large
```

**GCP (GKE):**
```bash
gcloud container clusters create loan-prediction \
  --num-nodes=3 \
  --machine-type=n1-standard-2 \
  --zone=us-central1-a
```

### Step 2: Install Ingress Controller
```bash
# Nginx Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.9.0/deploy/static/provider/cloud/deploy.yaml

# Verify
kubectl get pods -n ingress-nginx
```

### Step 3: Deploy Application
```bash
# Navigate to k8s directory
cd k8s/

# Apply all manifests in order
kubectl apply -f namespace.yaml
kubectl apply -f pvc.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f deployment-redis.yaml
kubectl apply -f service-redis.yaml
kubectl apply -f deployment-mlflow.yaml
kubectl apply -f service-mlflow.yaml
kubectl apply -f deployment-api.yaml
kubectl apply -f service-api.yaml
kubectl apply -f deployment-celery-worker.yaml
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml
```

### Step 4: Verify Deployment
```bash
# Check all resources
kubectl get all -n loan-default-prediction

# Expected output:
# - 1 Redis pod (Running)
# - 1 MLflow pod (Running)
# - 3 API pods (Running)
# - 2 Celery worker pods (Running)

# Check pod status
kubectl get pods -n loan-default-prediction

# Check logs
kubectl logs -n loan-default-prediction deployment/loan-api
kubectl logs -n loan-default-prediction deployment/mlflow-server
kubectl logs -n loan-default-prediction deployment/celery-worker

# Check ingress
kubectl get ingress -n loan-default-prediction
```

### Step 5: Test Endpoints
```bash
# Get Ingress IP
kubectl get ingress -n loan-default-prediction

# Test API health
curl http://loan-api.yourdomain.com/healthz

# Access MLflow UI
open http://mlflow.yourdomain.com
```

---

## 🔍 What's New: MLflow Server

### Before (File-based):
```
ConfigMap: MLFLOW_TRACKING_URI: "file:///app/mlflow"
```
- ❌ No web UI
- ❌ Can't track experiments remotely
- ❌ Limited to local file storage

### After (Server-based):
```
ConfigMap: MLFLOW_TRACKING_URI: "http://mlflow-service:5000"
```
- ✅ Full MLflow UI at `mlflow.yourdomain.com`
- ✅ Remote experiment tracking
- ✅ Model registry with versioning
- ✅ Centralized artifact storage
- ✅ Multi-user collaboration

### MLflow UI Features:
1. **Experiments Dashboard:** View all training runs, compare metrics
2. **Model Registry:** Manage model versions (Staging → Production)
3. **Artifacts Browser:** Download models, plots, data
4. **Comparison Tools:** Side-by-side metric comparison
5. **REST API:** Programmatic access to models

---

## 🎯 How to Explain This in an Interview

### "Walk me through your Kubernetes architecture"

**Answer Template:**

*"I designed a microservices-based ML deployment on Kubernetes with several key components:*

#### 1. **Core Services:**
- **FastAPI application** running 3 replicas behind a LoadBalancer for high availability
- **Redis** serving as a message broker for asynchronous batch predictions via Celery
- **Celery workers** (2 replicas) handling long-running batch inference jobs

#### 2. **ML Operations:**
- Deployed a **dedicated MLflow server** for experiment tracking and model registry
- Used a **PersistentVolumeClaim** (10Gi) for shared access to models between API and workers
- Configured the MLflow tracking URI to point to the centralized server, enabling:
  - Team collaboration on experiments
  - Model versioning and staging workflow
  - Centralized artifact storage

#### 3. **Scalability & Reliability:**
- Implemented **Horizontal Pod Autoscaling (HPA)** to scale API pods from 3 to 10 based on:
  - CPU utilization (70% threshold)
  - Memory utilization (80% threshold)
- Configured smart scaling policies:
  - Fast scale-up (100% increase every 30s)
  - Slow scale-down (50% decrease with 5-min stabilization)
- Added liveness and readiness probes for automatic health checks

#### 4. **Networking & Routing:**
- Used an **Ingress controller** with domain-based routing:
  - `loan-api.domain.com` → FastAPI service
  - `mlflow.domain.com` → MLflow UI
- Configured TLS termination at the Ingress level
- Added rate limiting (100 req/min) to prevent abuse

#### 5. **Security Best Practices:**
- Separated sensitive credentials (API keys) into Kubernetes Secrets
- Stored non-sensitive config in ConfigMaps for easy updates
- Used RBAC and namespaces for isolation
- Implemented TLS/SSL for all external communication

#### 6. **Resource Management:**
- Set appropriate resource requests and limits for each service:
  - API: 512Mi-1Gi RAM, 250m-500m CPU
  - Workers: 768Mi-2Gi RAM (more for ML workloads)
  - Redis: 128Mi-256Mi RAM
- This ensures efficient cluster utilization and prevents resource starvation"

---

### "Why did you choose this architecture?"

**Key Points to Mention:**

1. **Separation of concerns:** API handles real-time requests, Celery handles batch jobs
2. **Scalability:** HPA allows automatic scaling during traffic spikes
3. **Observability:** MLflow provides centralized tracking and model governance
4. **Fault tolerance:** Multiple replicas ensure zero downtime
5. **Cost efficiency:** Resource limits prevent over-provisioning

---

### "What would you improve?"

**Honest answers:**

1. **Monitoring:** Add Prometheus + Grafana for metrics
2. **Logging:** Implement ELK/EFK stack for centralized logging
3. **CI/CD:** Automate deployments with GitOps (ArgoCD, Flux)
4. **Storage:** Migrate MLflow artifacts to S3/GCS for better performance
5. **Database:** Use managed PostgreSQL (RDS, Cloud SQL) instead of in-cluster
6. **Security:** Implement Pod Security Policies, network policies
7. **Cost optimization:** Use spot/preemptible instances for workers

---

## 🔄 Common Operations

### Update Configuration
```bash
# Edit configmap
kubectl edit configmap loan-api-config -n loan-default-prediction

# Restart pods to pick up changes
kubectl rollout restart deployment/loan-api -n loan-default-prediction
kubectl rollout restart deployment/celery-worker -n loan-default-prediction
```

### Scale Manually
```bash
# Scale API
kubectl scale deployment loan-api --replicas=5 -n loan-default-prediction

# Scale workers
kubectl scale deployment celery-worker --replicas=4 -n loan-default-prediction
```

### View Logs
```bash
# Live tail
kubectl logs -f deployment/loan-api -n loan-default-prediction

# Last 100 lines
kubectl logs --tail=100 deployment/mlflow-server -n loan-default-prediction
```

### Exec into Pod
```bash
# Get pod name
kubectl get pods -n loan-default-prediction

# Shell into pod
kubectl exec -it <pod-name> -n loan-default-prediction -- /bin/bash
```

### Rolling Update
```bash
# Update image
kubectl set image deployment/loan-api api=your-registry/loan-default-api:v2.0 -n loan-default-prediction

# Check rollout status
kubectl rollout status deployment/loan-api -n loan-default-prediction

# Rollback if needed
kubectl rollout undo deployment/loan-api -n loan-default-prediction
```

---

## 🐛 Troubleshooting

### Pods Not Starting
```bash
# Check pod status
kubectl describe pod <pod-name> -n loan-default-prediction

# Common issues:
# - ImagePullBackOff: Wrong image name/registry
# - CrashLoopBackOff: App crashes on startup (check logs)
# - Pending: Insufficient cluster resources
```

### Ingress Not Working
```bash
# Check ingress status
kubectl describe ingress loan-prediction-ingress -n loan-default-prediction

# Verify ingress controller is running
kubectl get pods -n ingress-nginx

# Check DNS
nslookup loan-api.yourdomain.com
```

### MLflow Can't Access Models
```bash
# Check PVC is bound
kubectl get pvc -n loan-default-prediction

# Verify mount in pod
kubectl exec -it <api-pod> -n loan-default-prediction -- ls -la /app/mlflow
```

### High Memory Usage
```bash
# Check resource usage
kubectl top pods -n loan-default-prediction

# Increase limits in deployment YAML
resources:
  limits:
    memory: "2Gi"  # Increase as needed
```

---

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [MLflow Tracking Server](https://mlflow.org/docs/latest/tracking.html#tracking-server)
- [Nginx Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [Horizontal Pod Autoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

---

## ✅ Pre-Deployment Checklist

- [ ] Docker image built and pushed to registry
- [ ] Image references updated in deployment files
- [ ] Secrets updated with real credentials (not defaults)
- [ ] Ingress domains updated with your domain
- [ ] TLS certificates configured
- [ ] Kubernetes cluster running and accessible
- [ ] Ingress controller installed
- [ ] kubectl configured with correct context
- [ ] Namespace created
- [ ] Storage class available for PVCs

---

## 🎓 Learning Path

1. **Start Simple:** Deploy to Minikube locally
2. **Add Complexity:** Enable MLflow server
3. **Go Production:** Add PostgreSQL, external storage
4. **Monitor:** Add Prometheus + Grafana
5. **Automate:** Set up CI/CD pipeline
6. **Secure:** Implement RBAC, network policies, Pod Security

Good luck with your deployment! 🚀
