"""
Print Cost FastAPI Application

Production-ready FastAPI service for PDF printing cost calculation.
Follows ML engineering best practices with proper error handling,
logging, and monitoring.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import logging
from contextlib import asynccontextmanager
import pymupdf

from src.services.cost_calculator import CostCalculator
from src.services.model_manager import ModelManager
from src.exceptions import (
    InvalidPDFError,
    PredictionError,
    ModelLoadError,
    PageCountError,
    FileSizeError,
)
from src.config import (
    DEFAULT_DPI,
    MAX_FILE_SIZE_MB,
    MIN_PAGES,
    MAX_PAGES,
    MODEL_VERSION,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Response models
class PriceDetail(BaseModel):
    """Price breakdown by category."""
    price: int = Field(..., description="Price per page in IDR")
    pages: int = Field(..., ge=0, description="Number of pages in this category")
    subtotal: int = Field(..., ge=0, description="Subtotal for this category")
    category: str = Field(..., description="Print category name")


class CostCalculationResponse(BaseModel):
    """Response model for cost calculation."""
    total_pages: int = Field(..., ge=1, description="Total number of pages")
    total_price: int = Field(..., ge=0, description="Total printing cost in IDR")
    details: List[PriceDetail] = Field(..., description="Breakdown by category")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    ml_model_version: str = Field(default=MODEL_VERSION, description="ML model version used")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    ml_model_version: str
    ml_model_loaded: bool


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: str
    error_type: str


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup: Preload model
    logger.info("Starting Print Cost API...")
    try:
        ModelManager.get_default_model()
        logger.info("Model loaded successfully")
    except ModelLoadError as e:
        logger.error(f"Failed to load model on startup: {e}")
    
    yield  # Application runs
    
    # Shutdown: Cleanup
    logger.info("Shutting down Print Cost API...")
    ModelManager.clear_cache()


# Initialize FastAPI app
app = FastAPI(
    title="Print Cost API",
    description="ML-powered PDF printing cost calculator with 99% F1 score accuracy",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize cost calculator (lazy loaded with model caching)
calculator = None


def get_calculator() -> CostCalculator:
    """Get or create cost calculator instance."""
    global calculator
    if calculator is None:
        calculator = CostCalculator()
    return calculator


# Exception handlers
@app.exception_handler(InvalidPDFError)
async def invalid_pdf_handler(request, exc: InvalidPDFError):
    """Handle invalid PDF errors."""
    logger.warning(f"Invalid PDF: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Invalid PDF",
            detail=str(exc),
            error_type="InvalidPDFError"
        ).dict()
    )


@app.exception_handler(PageCountError)
async def page_count_handler(request, exc: PageCountError):
    """Handle page count validation errors."""
    logger.warning(f"Page count error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Invalid page count",
            detail=str(exc),
            error_type="PageCountError"
        ).dict()
    )


@app.exception_handler(FileSizeError)
async def file_size_handler(request, exc: FileSizeError):
    """Handle file size errors."""
    logger.warning(f"File size error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content=ErrorResponse(
            error="File too large",
            detail=str(exc),
            error_type="FileSizeError"
        ).dict()
    )


@app.exception_handler(PredictionError)
async def prediction_error_handler(request, exc: PredictionError):
    """Handle prediction errors."""
    logger.error(f"Prediction error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Prediction failed",
            detail=str(exc),
            error_type="PredictionError"
        ).dict()
    )


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Print Cost API - ML-powered printing cost calculator",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "calculate_cost": "POST /api/v1/calculate-cost",
        }
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns service health status and model information.
    """
    try:
        # Check if model is loaded
        model = ModelManager.get_default_model()
        model_loaded = model is not None
        
        return HealthCheckResponse(
            status="healthy",
            ml_model_version=MODEL_VERSION,
            ml_model_loaded=model_loaded
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=HealthCheckResponse(
                status="unhealthy",
                ml_model_version=MODEL_VERSION,
                ml_model_loaded=False
            ).dict()
        )


@app.post(
    "/api/v1/calculate-cost",
    response_model=CostCalculationResponse,
    tags=["Cost Calculation"],
    status_code=status.HTTP_200_OK,
    summary="Calculate PDF printing cost",
    description="Upload a PDF file to get accurate printing cost breakdown by color category"
)
async def calculate_print_cost(
    file: UploadFile = File(..., description="PDF file to analyze"),
    dpi: int = DEFAULT_DPI
) -> CostCalculationResponse:
    """
    Calculate printing cost for a PDF document.
    
    **Process**:
    1. Validates PDF file format and size
    2. Extracts CMYK color features from each page
    3. Predicts print category using ML model (99% F1 score)
    4. Calculates cost breakdown
    
    **Returns**:
    - Total pages and price
    - Breakdown by category (Mono, Color Light, Color Standard, etc.)
    - Processing time
    
    **Raises**:
    - 400: Invalid PDF or file validation failed
    - 413: File too large (>50MB)
    - 500: Prediction failed
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )
    
    # Read and validate file size
    pdf_bytes = await file.read()
    file_size_mb = len(pdf_bytes) / (1024 * 1024)
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise FileSizeError(file_size_mb, MAX_FILE_SIZE_MB)
    
    logger.info(f"Processing PDF: {file.filename} ({file_size_mb:.2f} MB)")
    
    try:
        # Open PDF document
        pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
        # Validate page count
        page_count = len(pdf)
        if not (MIN_PAGES <= page_count <= MAX_PAGES):
            pdf.close()
            raise PageCountError(page_count, MIN_PAGES, MAX_PAGES)
        
        # Calculate cost
        calc = get_calculator()
        result = calc.calculate_cost(pdf, dpi=dpi)
        
        pdf.close()
        
        logger.info(
            f"Successfully processed {result['total_pages']} pages "
            f"in {result['processing_time']:.2f}s - Total: IDR {result['total_price']:,}"
        )
        
        # Add model version to response
        result['ml_model_version'] = MODEL_VERSION
        
        return CostCalculationResponse(**result)
        
    except InvalidPDFError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted PDF: {str(e)}"
        )
    except (PageCountError, FileSizeError):
        # Re-raise our custom exceptions (handled by exception handlers)
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing PDF: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {str(e)}"
        )


@app.get("/api/v1/model-info", tags=["Model"])
async def get_model_info():
    """
    Get information about the ML model.
    
    Returns model metadata, cache status, and configuration.
    """
    cache_info = ModelManager.get_cache_info()
    
    return {
        "model_version": MODEL_VERSION,
        "model_type": "XGBoost Classifier",
        "f1_score": 0.99,
        "features": ["cmy", "k", "cmyk"],
        "categories": 5,
        "default_dpi": DEFAULT_DPI,
        "cache_info": cache_info,
        "limits": {
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "min_pages": MIN_PAGES,
            "max_pages": MAX_PAGES,
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main-fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )