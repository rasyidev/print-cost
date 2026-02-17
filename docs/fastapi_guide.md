# FastAPI Best Practices Implementation - Quick Reference

## What Was Improved

### ❌ Before (16 lines, basic)
```python
@app.post("/print-cost/")
async def calculate_print_cost(file: UploadFile):
    pdf = pymupdf.open(stream=file.file.read())
    pc = PrintCost(pdf, "models/xgboost_98.64_cmy_k_cmyk_7_dpi.pkl")
    result = pc.predict(dpi=7)
    return {"result": result}
```

Problems:
- No error handling
- No input validation
- Hardcoded model path
- No logging
- No API documentation
- No health checks
- No response models

### ✅ After (340+ lines, production-ready)

**Key Improvements:**

1. **Pydantic Models** - Type-safe request/response
2. **Error Handling** - Custom exception handlers for all error types
3. **Logging** - Structured logging with context
4. **Health Checks** - `/health` endpoint for monitoring
5. **API Documentation** - Auto-generated OpenAPI docs
6. **CORS Support** - For web clients
7. **Lifespan Management** - Proper startup/shutdown
8. **Model Caching** - Preload model on startup
9. **Input Validation** - File size, type, page count checks
10. **Versioning** - `/api/v1/` prefix for API versioning

---

## API Endpoints

### 1. Root - `GET /`
```bash
curl http://localhost:8000/
```
Returns API information and available endpoints.

### 2. Health Check - `GET /health`
```bash
curl http://localhost:8000/health
```
Returns:
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "model_loaded": true
}
```

### 3. Calculate Cost - `POST /api/v1/calculate-cost`
```bash
curl -X POST "http://localhost:8000/api/v1/calculate-cost" \
  -F "file=@document.pdf" \
  -F "dpi=7"
```

Returns:
```json
{
  "total_pages": 10,
  "total_price": 7500,
  "details": [
    {
      "price": 500,
      "pages": 5,
      "subtotal": 2500,
      "category": "Mono Print"
    },
    {
      "price": 1000,
      "pages": 5,
      "subtotal": 5000,
      "category": "Color Standard"
    }
  ],
  "processing_time": 0.85,
  "model_version": "1.0.0"
}
```

### 4. Model Info - `GET /api/v1/model-info`
```bash
curl http://localhost:8000/api/v1/model-info
```
Returns model metadata, features, and configuration.

---

## Running the API

### Development Mode
```bash
cd /Users/rasyidev/Documents/rasyidevcode/print-cost

# Run with auto-reload
python main-fastapi.py

# Or using uvicorn directly
uvicorn main-fastapi:app --reload --port 8000
```

### Production Mode
```bash
# Use gunicorn with uvicorn workers
gunicorn main-fastapi:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Access Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Error Handling

All errors return structured responses:

```json
{
  "error": "Invalid PDF",
  "detail": "Cannot open PDF file: corrupted",
  "error_type": "InvalidPDFError"
}
```

**Error Codes:**
- `400` - Bad Request (invalid PDF, wrong file type, page count violations)
- `413` - File Too Large (>50MB)
- `500` - Internal Server Error (prediction failure)
- `503` - Service Unavailable (health check failed)

---

## Best Practices Implemented

### 1. **Dependency Injection**
```python
def get_calculator() -> CostCalculator:
    """Lazy loading with caching."""
    global calculator
    if calculator is None:
        calculator = CostCalculator()
    return calculator
```

### 2. **Lifespan Events**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Preload model
    ModelManager.get_default_model()
    yield
    # Shutdown: Clear cache
    ModelManager.clear_cache()
```

### 3. **Exception Handlers**
```python
@app.exception_handler(InvalidPDFError)
async def invalid_pdf_handler(request, exc):
    return JSONResponse(status_code=400, content={...})
```

### 4. **Response Models**
```python
class CostCalculationResponse(BaseModel):
    total_pages: int = Field(..., ge=1)
    total_price: int = Field(..., ge=0)
    details: List[PriceDetail]
    processing_time: float
```

### 5. **Comprehensive Documentation**
- Detailed docstrings for all endpoints
- Parameter descriptions
- Example responses
- Error documentation

---

## Testing the API

### Using cURL
```bash
# Health check
curl http://localhost:8000/health

# Calculate cost
curl -X POST http://localhost:8000/api/v1/calculate-cost \
  -F "file=@test.pdf" \
  -F "dpi=7"
```

### Using Python requests
```python
import requests

# Upload PDF
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/calculate-cost",
        files={"file": f},
        data={"dpi": 7}
    )

result = response.json()
print(f"Total: IDR {result['total_price']:,}")
```

### Using httpie
```bash
http POST http://localhost:8000/api/v1/calculate-cost \
  file@document.pdf dpi=7
```

---

## Monitoring & Logging

### Logs Format
```
2024-02-17 16:21:00 - main-fastapi - INFO - Processing PDF: document.pdf (2.34 MB)
2024-02-17 16:21:01 - main-fastapi - INFO - Successfully processed 10 pages in 0.85s - Total: IDR 7,500
```

### Integrate with Monitoring
The `/health` endpoint can be used with:
- Kubernetes liveness/readiness probes
- Docker health checks
- Load balancer health checks
- Monitoring systems (Prometheus, Datadog, etc.)

---

## Migration from Old Code

### Step 1: Stop using old endpoint
```python
# OLD (deprecated)
@app.post("/print-cost/")

# NEW (versioned)
@app.post("/api/v1/calculate-cost")
```

### Step 2: Update clients
Update any Streamlit or frontend code to use new endpoint:
```python
# Old
requests.post("http://localhost:8000/print-cost/", ...)

# New
requests.post("http://localhost:8000/api/v1/calculate-cost", ...)
```

---

## Summary

✅ **Removed**:
- Flask templates (`src/templates/`)
- Old Flask app (`src/flask_old_main.py`)

✅ **Added**:
- Pydantic models for type safety
- Comprehensive error handling
- Health check endpoint
- Model info endpoint
- API versioning (`/api/v1/`)
- CORS support
- Logging
- Auto-generated documentation
- Input validation
- Response models

🎯 **Result**: Production-ready FastAPI service following ML engineering best practices!
