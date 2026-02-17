"""
Legacy PrintCost class for backward compatibility.

This module provides the original PrintCost class interface while delegating
to the new service layer architecture. This maintains backward compatibility
with existing code while enabling a clean migration path.
"""

import pickle
import time
from typing import Dict, Any
import pandas as pd
import numpy as np
import pymupdf
from PIL import Image
import io

from .config import ROOT_DIR
from .services.image_processor import ImageProcessor
from .services.model_manager import ModelManager


class PrintCost:
    """
    Legacy PrintCost class maintained for backward compatibility.
    
    This class now delegates to the new service layer architecture while
    maintaining the original API interface for existing code.
    
    Attributes:
        file_path: Path to PDF file or PyMuPDF Document object
        model_pkl_path: Path to the pickled model file
        model: Loaded machine learning model
        df: DataFrame containing extracted CMYK features and predictions
    """
    
    def __init__(self, file_path: str | pymupdf.Document, model_pkl_path: str) -> None:
        """
        Initialize PrintCost calculator.
        
        Args:
            file_path: Path to PDF file or PyMuPDF Document object
            model_pkl_path: Path to the pickled model file
        """
        self.file_path = file_path
        self.model_pkl_path = model_pkl_path
        self.model = ModelManager.load_model(self.model_pkl_path)
        self.image_processor = ImageProcessor()
        self.df: pd.DataFrame = None
    
    def extract_cmyk(self, dpi: int = 7) -> pd.DataFrame:
        """
        Extract CMYK features from all pages of a PDF.
        
        Args:
            dpi: Resolution for PDF rendering (default: 7)
            
        Returns:
            DataFrame with columns: c, m, y, k, cmy, cmyk
        """
        # Reset dataframe for new extraction
        cmyk_data = {
            "c": [],
            "m": [],
            "y": [],
            "k": [],
            "cmy": [],
            "cmyk": [],
        }
        
        # Open PDF if path provided
        pdf_obj = (
            pymupdf.open(self.file_path) 
            if isinstance(self.file_path, str) 
            else self.file_path
        )
        
        # Extract CMYK from each page
        for page in pdf_obj:
            pixmap = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pixmap.tobytes()))
            
            # Use ImageProcessor service
            c, m, y, k = self.image_processor.calculate_cmyk_percentage(img)
            cmy = c + m + y
            cmyk = cmy + k
            
            cmyk_data["c"].append(c)
            cmyk_data["m"].append(m)
            cmyk_data["y"].append(y)
            cmyk_data["k"].append(k)
            cmyk_data["cmy"].append(cmy)
            cmyk_data["cmyk"].append(cmyk)
        
        self.df = pd.DataFrame(cmyk_data)
        return self.df
    
    def predict(self, dpi: int = 7) -> Dict[str, Any]:
        """
        Predict printing costs for the PDF.
        
        Args:
            dpi: Resolution for PDF rendering (default: 7)
            
        Returns:
            Dictionary containing:
                - total_pages: Total number of pages
                - total_price: Total printing cost in IDR
                - details: List of dicts with per-category breakdown
        """
        start_time = time.time()
        
        # Extract features
        self.extract_cmyk(dpi=dpi)
        
        # Make predictions
        y_pred = self.model.predict(self.df[["cmy", "k", "cmyk"]])
        self.df["price"] = y_pred
        
        # Map label indices to prices
        from .config import PRICE_LABEL_MAP
        self.df["price"] = self.df["price"].replace(PRICE_LABEL_MAP)
        
        # Generate response
        response = self._generate_response(start_time)
        return response
    
    def _generate_response(self, start_time: float) -> Dict[str, Any]:
        """
        Generate formatted response with cost breakdown.
        
        Args:
            start_time: Timestamp when processing started
            
        Returns:
            Dictionary with cost breakdown and metadata
        """
        from .config import PRICE_CATEGORY_MAP
        
        # Aggregate by price category
        result_df = (
            self.df["price"]
            .value_counts()
            .reset_index()
            .sort_values("price")
            .rename(columns={"count": "pages"})
        )
        
        # Calculate subtotals
        result_df["subtotal"] = result_df["price"] * result_df["pages"]
        
        # Add category names
        result_df["category"] = result_df["price"].map(PRICE_CATEGORY_MAP)
        
        return {
            "total_pages": int(result_df["pages"].sum()),
            "total_price": int(result_df["subtotal"].sum()),
            "details": result_df.to_dict(orient="records"),
        }
