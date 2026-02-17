"""
Cost calculation service for print pricing.

This module orchestrates the entire cost calculation workflow including
PDF processing, feature extraction, prediction, and result formatting.
"""

import time
from typing import Any, Dict
import logging
import pandas as pd
import pymupdf
from PIL import Image
import io

from .image_processor import ImageProcessor
from .model_manager import ModelManager
from ..exceptions import InvalidPDFError, PredictionError, PageCountError
from ..config import (
    PRICE_LABEL_MAP,
    PRICE_CATEGORY_MAP,
    DEFAULT_DPI,
    MODEL_FEATURES,
    MIN_PAGES,
    MAX_PAGES,
)

logger = logging.getLogger(__name__)


class CostCalculator:
    """
    Orchestrate PDF processing and cost calculation.
    
    This service coordinates the entire workflow from PDF input to
    cost breakdown output.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the cost calculator.
        
        Args:
            model_path: Optional path to model file. Uses default if not provided.
        """
        self.image_processor = ImageProcessor()
        self.model = ModelManager.get_default_model() if model_path is None else ModelManager.load_model(model_path)
        logger.info("CostCalculator initialized successfully")
    
    def calculate_cost(
        self, 
        pdf_file: pymupdf.Document | str, 
        dpi: int = DEFAULT_DPI
    ) -> Dict[str, Any]:
        """
        Calculate printing cost for a PDF document.
        
        Args:
            pdf_file: PyMuPDF Document object or path to PDF file
            dpi: Resolution for rendering PDF pages (default from config)
            
        Returns:
            Dictionary containing:
                - total_pages: Total number of pages
                - total_price: Total printing cost in IDR
                - details: List of dictionaries with per-category breakdown
                - processing_time: Time taken for calculation in seconds
                
        Raises:
            InvalidPDFError: If PDF cannot be opened or processed
            PageCountError: If page count is outside valid range
            PredictionError: If model prediction fails
        """
        start_time = time.time()
        
        try:
            # Open PDF if path provided
            pdf = self._open_pdf(pdf_file)
            
            # Validate page count
            page_count = len(pdf)
            if not (MIN_PAGES <= page_count <= MAX_PAGES):
                raise PageCountError(page_count, MIN_PAGES, MAX_PAGES)
            
            logger.info(f"Processing PDF with {page_count} pages at {dpi} DPI")
            
            # Extract CMYK features from all pages
            df = self._extract_features(pdf, dpi)
            
            # Predict prices for all pages
            predictions = self._predict_prices(df)
            
            # Generate response
            processing_time = time.time() - start_time
            result = self._generate_response(predictions, processing_time)
            
            logger.info(
                f"Cost calculation complete: {result['total_pages']} pages, "
                f"IDR {result['total_price']:,} in {processing_time:.2f}s"
            )
            
            return result
            
        except (InvalidPDFError, PageCountError, PredictionError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in cost calculation: {str(e)}")
            raise PredictionError("Cost calculation failed", e)
    
    def _open_pdf(self, pdf_file: pymupdf.Document | str) -> pymupdf.Document:
        """Open PDF file and validate it."""
        if isinstance(pdf_file, pymupdf.Document):
            return pdf_file
        
        try:
            return pymupdf.open(pdf_file)
        except Exception as e:
            logger.error(f"Failed to open PDF: {str(e)}")
            raise InvalidPDFError(f"Cannot open PDF file: {str(e)}")
    
    def _extract_features(self, pdf: pymupdf.Document, dpi: int) -> pd.DataFrame:
        """
        Extract CMYK features from all pages of a PDF.
        
        Args:
            pdf: PyMuPDF Document object
            dpi: Resolution for rendering
            
        Returns:
            DataFrame with columns: c, m, y, k, cmy, cmyk
        """
        features = {
            "c": [],
            "m": [],
            "y": [],
            "k": [],
            "cmy": [],
            "cmyk": [],
        }
        
        for page_num, page in enumerate(pdf, start=1):
            try:
                # Render page to image
                pixmap = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pixmap.tobytes()))
                
                # Calculate CMYK percentages
                c, m, y, k = self.image_processor.calculate_cmyk_percentage(img)
                cmy = c + m + y
                cmyk = cmy + k
                
                # Append to features
                features["c"].append(c)
                features["m"].append(m)
                features["y"].append(y)
                features["k"].append(k)
                features["cmy"].append(cmy)
                features["cmyk"].append(cmyk)
                
                logger.debug(f"Page {page_num}: C={c:.2f}, M={m:.2f}, Y={y:.2f}, K={k:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                raise PredictionError(f"Failed to extract features from page {page_num}", e)
        
        return pd.DataFrame(features)
    
    def _predict_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict prices for all pages using the ML model.
        
        Args:
            df: DataFrame with CMYK features
            
        Returns:
            DataFrame with added 'price' column
        """
        try:
            # Select features for prediction
            X = df[MODEL_FEATURES]
            
            # Get predictions (label indices)
            y_pred = self.model.predict(X)
            
            # Map label indices to actual prices
            df["price"] = pd.Series(y_pred).map(PRICE_LABEL_MAP)
            
            return df
            
        except Exception as e:
            logger.error(f"Model prediction failed: {str(e)}")
            raise PredictionError("Model prediction failed", e)
    
    def _generate_response(self, df: pd.DataFrame, processing_time: float) -> Dict[str, Any]:
        """
        Generate formatted response with cost breakdown.
        
        Args:
            df: DataFrame with predictions
            processing_time: Time taken for processing
            
        Returns:
            Formatted response dictionary
        """
        # Aggregate by price category
        result_df = (
            df["price"]
            .value_counts()
            .reset_index()
            .sort_values("price")
            .rename(columns={"count": "pages"})
        )
        
        # Calculate subtotals
        result_df["subtotal"] = result_df["price"] * result_df["pages"]
        
        # Add category names
        result_df["category"] = result_df["price"].map(PRICE_CATEGORY_MAP)
        
        # Build response
        return {
            "total_pages": int(result_df["pages"].sum()),
            "total_price": int(result_df["subtotal"].sum()),
            "details": result_df.to_dict(orient="records"),
            "processing_time": round(processing_time, 2),
        }
