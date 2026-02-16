# API Examples

Complete examples for the Loan Default Prediction API with real request/response pairs.

## Table of Contents
- [Authentication](#authentication)
- [Single Prediction](#single-prediction)
- [Batch Prediction](#batch-prediction)
- [Model Information](#model-information)
- [Health Checks](#health-checks)
- [Error Cases](#error-cases)

---

## Authentication

All protected endpoints require an API key in the request header:

```bash
X-API-Key: your-secret-api-key-here-change-in-production
```

**Note**: Change the API key in your `.env` file for production use.

---

## Single Prediction

### Example 1: Low Risk Customer (No Default)

**Request:**
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

**Response (200 OK):**
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

**Interpretation:**
- `prediction: 0` → Customer will NOT default
- `probability: 0.0823` → 8.23% chance of default
- `default_risk: "Low"` → Low risk customer

---

### Example 2: High Risk Customer (Potential Default)

**Request:**
```bash
curl -X POST "http://localhost:8005/api/v1/predict" \
  -H "X-API-Key: your-secret-api-key-here-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "employed": 0,
    "bank_balance": 500.0,
    "annual_salary": 25000.0
  }'
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "prediction": 1,
    "probability": 0.8567,
    "default_risk": "High",
    "model_version": "1",
    "features_used": {
      "employed": 0,
      "bank_balance": 500.0,
      "annual_salary": 25000.0
    }
  },
  "message": "Prediction completed successfully"
}
```

**Interpretation:**
- `prediction: 1` → Customer WILL default
- `probability: 0.8567` → 85.67% chance of default
- `default_risk: "High"` → High risk customer

---

### Example 3: Medium Risk Customer

**Request:**
```bash
curl -X POST "http://localhost:8005/api/v1/predict" \
  -H "X-API-Key: your-secret-api-key-here-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "employed": 1,
    "bank_balance": 5000.0,
    "annual_salary": 35000.0
  }'
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "prediction": 0,
    "probability": 0.4521,
    "default_risk": "Medium",
    "model_version": "1",
    "features_used": {
      "employed": 1,
      "bank_balance": 5000.0,
      "annual_salary": 35000.0
    }
  },
  "message": "Prediction completed successfully"
}
```

**Interpretation:**
- `prediction: 0` → Customer will NOT default (but borderline)
- `probability: 0.4521` → 45.21% chance of default
- `default_risk: "Medium"` → Medium risk customer

---

## Batch Prediction

### Submit Batch Job

**Request:**
```bash
curl -X POST "http://localhost:8005/api/v1/predict/batch" \
  -H "X-API-Key: your-secret-api-key-here-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "employed": 1,
        "bank_balance": 10000.0,
        "annual_salary": 50000.0
      },
      {
        "employed": 0,
        "bank_balance": 2000.0,
        "annual_salary": 30000.0
      },
      {
        "employed": 1,
        "bank_balance": 15000.0,
        "annual_salary": 75000.0
      }
    ]
  }'
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "data": {
    "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "status": "PENDING",
    "message": "Batch job submitted successfully"
    "total_predictions": 3
  },
  "message": "Batch job submitted successfully"
}
```

---

### Check Batch Job Status (Pending)

**Request:**
```bash
curl -X GET "http://localhost:8005/api/v1/predict/batch/a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
  -H "X-API-Key: your-secret-api-key-here-change-in-production"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "status": "PENDING",
    "progress": 0.0
  },
  "message": "Job is pending"
}
```

---

### Check Batch Job Status (In Progress)

**Request:**
```bash
curl -X GET "http://localhost:8005/api/v1/predict/batch/a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
  -H "X-API-Key: your-secret-api-key-here-change-in-production"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "status": "PROGRESS",
    "progress": 66.67,
    "total": 3,
    "completed": 2
  },
  "message": "Job is in progress"
}
```

---

### Check Batch Job Status (Completed)

**Request:**
```bash
curl -X GET "http://localhost:8005/api/v1/predict/batch/a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
  -H "X-API-Key: your-secret-api-key-here-change-in-production"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "status": "SUCCESS",
    "progress": 100.0,
    "total": 3,
    "completed": 3,
    "results": [
      {
        "prediction": 0,
        "probability": 0.0823,
        "default_risk": "Low",
        "features_used": {
          "employed": 1,
          "bank_balance": 10000.0,
          "annual_salary": 50000.0
        }
      },
      {
        "prediction": 1,
        "probability": 0.7234,
        "default_risk": "High",
        "features_used": {
          "employed": 0,
          "bank_balance": 2000.0,
          "annual_salary": 30000.0
        }
      },
      {
        "prediction": 0,
        "probability": 0.0512,
        "default_risk": "Low",
        "features_used": {
          "employed": 1,
          "bank_balance": 15000.0,
          "annual_salary": 75000.0
        }
      }
    ]
  },
  "message": "Job completed successfully"
}
```

---

## Model Information

**Request:**
```bash
curl -X GET "http://localhost:8005/api/v1/model/info" \
  -H "X-API-Key: your-secret-api-key-here-change-in-production"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "model_name": "loan_default_model",
    "model_version": "1",
    "model_stage": "Production",
    "features": [
      "employed",
      "bank_balance",
      "annual_salary"
    ],
    "engineered_features": [
      "Saving_Rate"
    ],
    "model_type": "XGBClassifier",
    "created_at": "2026-02-15T14:30:00"
  },
  "message": "Model information retrieved successfully"
}
```

---

## Health Checks

### Liveness Probe

**Request:**
```bash
curl -X GET "http://localhost:8005/healthz"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "status": "healthy"
  },
  "message": "Service is healthy"
}
```

**Note:** No authentication required. Always returns 200 if service is running.

---

### Readiness Probe

**Request:**
```bash
curl -X GET "http://localhost:8005/readyz"
```

**Response (200 OK) - When Ready:**
```json
{
  "success": true,
  "data": {
    "status": "ready",
    "model_loaded": true,
    "model_name": "loan_default_model",
    "model_version": "1"
  },
  "message": "Service is ready"
}
```

**Response (503 Service Unavailable) - When Not Ready:**
```json
{
  "success": false,
  "data": {
    "status": "not_ready",
    "model_loaded": false
  },
  "message": "Model not loaded"
}
```

**Note:** No authentication required. Returns 503 if model not loaded.

---

## Error Cases

### 1. Missing API Key

**Request:**
```bash
curl -X POST "http://localhost:8005/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "employed": 1,
    "bank_balance": 10000.0,
    "annual_salary": 50000.0
  }'
```

**Response (422 Unprocessable Entity):**
```json
{
  "success": false,
  "error": "validation_error",
  "message": "X-API-Key header is required"
}
```

---

### 2. Invalid API Key

**Request:**
```bash
curl -X POST "http://localhost:8005/api/v1/predict" \
  -H "X-API-Key: wrong-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "employed": 1,
    "bank_balance": 10000.0,
    "annual_salary": 50000.0
  }'
```

**Response (403 Forbidden):**
```json
{
  "success": false,
  "error": "invalid_api_key",
  "message": "Invalid API key"
}
```

---

### 3. Invalid Input - Negative Values

**Request:**
```bash
curl -X POST "http://localhost:8005/api/v1/predict" \
  -H "X-API-Key: your-secret-api-key-here-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "employed": 1,
    "bank_balance": -1000.0,
    "annual_salary": 50000.0
  }'
```

**Response (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "bank_balance"],
      "msg": "Input should be greater than or equal to 0",
      "input": -1000.0,
      "ctx": {
        "ge": 0.0
      }
    }
  ]
}
```

---

### 4. Invalid Input - Out of Range Employment Status

**Request:**
```bash
curl -X POST "http://localhost:8005/api/v1/predict" \
  -H "X-API-Key: your-secret-api-key-here-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "employed": 2,
    "bank_balance": 10000.0,
    "annual_salary": 50000.0
  }'
```

**Response (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "employed"],
      "msg": "Input should be less than or equal to 1",
      "input": 2,
      "ctx": {
        "le": 1
      }
    }
  ]
}
```

---

### 5. Missing Required Field

**Request:**
```bash
curl -X POST "http://localhost:8005/api/v1/predict" \
  -H "X-API-Key: your-secret-api-key-here-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "employed": 1,
    "bank_balance": 10000.0
  }'
```

**Response (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "annual_salary"],
      "msg": "Field required",
      "input": {
        "employed": 1,
        "bank_balance": 10000.0
      }
    }
  ]
}
```

---

### 6. Rate Limit Exceeded

**Request:**
```bash
# After making >100 requests in 1 minute
curl -X POST "http://localhost:8005/api/v1/predict" \
  -H "X-API-Key: your-secret-api-key-here-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "employed": 1,
    "bank_balance": 10000.0,
    "annual_salary": 50000.0
  }'
```

**Response (429 Too Many Requests):**
```json
{
  "error": "Rate limit exceeded: 100 per 1 minute"
}
```

---

## Testing Locally

### Using the provided examples:

1. **Start the services:**
   ```bash
   docker-compose up
   ```

2. **Wait for services to be ready:**
   ```bash
   # Check health
   curl http://localhost:8005/healthz

   # Check readiness
   curl http://localhost:8005/readyz
   ```

3. **Make a test prediction:**
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

4. **View API documentation:**
   - Open browser: http://localhost:8005/docs
   - Interactive API testing available via Swagger UI

---

## Input Field Specifications

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `employed` | integer | Yes | 0 or 1 | Employment status (0=unemployed, 1=employed) |
| `bank_balance` | float | Yes | ≥ 0 | Current bank balance in dollars |
| `annual_salary` | float | Yes | > 0 | Annual salary in dollars |

## Output Field Specifications

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | integer | 0 = No default, 1 = Will default |
| `probability` | float | Probability of default (0.0 to 1.0) |
| `default_risk` | string | Risk level: "Low", "Medium", or "High" |
| `model_version` | string | Version of the deployed model |
| `features_used` | object | Echo of input features for verification |

## Risk Level Classification

- **Low Risk**: probability < 0.3
- **Medium Risk**: 0.3 ≤ probability < 0.7
- **High Risk**: probability ≥ 0.7

---

**For more information, see:**
- [README.md](README.md) - Full documentation
- [API Documentation](http://localhost:8005/docs) - Interactive API docs
- [DESIGN.md](DESIGN.md) - Architecture and design decisions
