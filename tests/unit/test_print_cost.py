"""
Unit tests for the legacy PrintCost class.

Tests the backward-compatible PrintCost class that delegates to services.
"""

import pytest
import pandas as pd
import pymupdf
import numpy as np

from src.helper import PrintCost
from src.services.model_manager import ModelManager


class TestPrintCostInit:
    """Test PrintCost initialization."""
    
    def test_init_with_file_path(self, sample_pdf_file, temp_model_file):
        """Test initialization with file path."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        
        assert pc.file_path == sample_pdf_file
        assert pc.model_pkl_path == temp_model_file
        assert pc.model is not None
        assert pc.image_processor is not None
    
    def test_init_with_pdf_document(self, sample_pdf_bytes, temp_model_file):
        """Test initialization with PyMuPDF Document object."""
        pdf_doc = pymupdf.open("pdf", sample_pdf_bytes)
        pc = PrintCost(pdf_doc, temp_model_file)
        
        assert pc.file_path == pdf_doc
        assert pc.model is not None
        pdf_doc.close()


class TestExtractCMYK:
    """Test CMYK extraction from PDF."""
    
    def test_extract_cmyk_returns_dataframe(self, sample_pdf_file, temp_model_file):
        """Test that extract_cmyk returns a DataFrame with correct structure."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        df = pc.extract_cmyk(dpi=7)
        
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['c', 'm', 'y', 'k', 'cmy', 'cmyk']
    
    def test_extract_cmyk_correct_number_of_rows(self, sample_pdf_file, temp_model_file):
        """Test that DataFrame has one row per page."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        df = pc.extract_cmyk(dpi=7)
        
        # Sample PDF has 3 pages
        assert len(df) == 3
    
    def test_extract_cmyk_values_in_valid_range(self, sample_pdf_file, temp_model_file):
        """Test that CMYK values are in valid percentage range."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        df = pc.extract_cmyk(dpi=7)
        
        # Individual CMYK components should be 0-100
        for col in ['c', 'm', 'y', 'k']:
            assert df[col].min() >= 0
            assert df[col].max() <= 100
    
    def test_extract_cmyk_calculated_fields(self, sample_pdf_file, temp_model_file):
        """Test that cmy and cmyk fields are correctly calculated."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        df = pc.extract_cmyk(dpi=7)
        
        for idx, row in df.iterrows():
            expected_cmy = row['c'] + row['m'] + row['y']
            expected_cmyk = expected_cmy + row['k']
            
            assert abs(row['cmy'] - expected_cmy) < 0.01
            assert abs(row['cmyk'] - expected_cmyk) < 0.01


class TestPredict:
    """Test prediction functionality."""
    
    def test_predict_returns_dict(self, sample_pdf_file, temp_model_file):
        """Test that predict returns a dictionary with expected keys."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        result = pc.predict(dpi=7)
        
        assert isinstance(result, dict)
        assert 'total_pages' in result
        assert 'total_price' in result
        assert 'details' in result
    
    def test_predict_total_pages_matches_pdf(self, sample_pdf_file, temp_model_file):
        """Test that total_pages matches the actual PDF page count."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        result = pc.predict(dpi=7)
        
        # Sample PDF has 3 pages
        assert result['total_pages'] == 3
    
    def test_predict_prices_are_integers(self, sample_pdf_file, temp_model_file):
        """Test that prices are returned as integers."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        result = pc.predict(dpi=7)
        
        assert isinstance(result['total_price'], int)
        for detail in result['details']:
            assert isinstance(detail['price'], (int, np.int64))
            assert isinstance(detail['subtotal'], (int, np.int64))
    
    def test_predict_details_structure(self, sample_pdf_file, temp_model_file):
        """Test that details list has correct structure."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        result = pc.predict(dpi=7)
        
        assert isinstance(result['details'], list)
        assert len(result['details']) > 0
        
        for detail in result['details']:
            assert 'price' in detail
            assert 'pages' in detail
            assert 'subtotal' in detail
            assert 'category' in detail
    
    def test_predict_subtotal_calculation(self, sample_pdf_file, temp_model_file):
        """Test that subtotals are correctly calculated."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        result = pc.predict(dpi=7)
        
        for detail in result['details']:
            expected_subtotal = detail['price'] * detail['pages']
            assert detail['subtotal'] == expected_subtotal
    
    def test_predict_total_price_sum(self, sample_pdf_file, temp_model_file):
        """Test that total_price is sum of all subtotals."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        result = pc.predict(dpi=7)
        
        expected_total = sum(detail['subtotal'] for detail in result['details'])
        assert result['total_price'] == expected_total
    
    def test_predict_total_pages_sum(self, sample_pdf_file, temp_model_file):
        """Test that total_pages is sum of all pages in details."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        result = pc.predict(dpi=7)
        
        expected_pages = sum(detail['pages'] for detail in result['details'])
        assert result['total_pages'] == expected_pages


class TestPrintCostIntegration:
    """Integration tests for PrintCost class."""
    
    def test_multiple_predictions_same_result(self, sample_pdf_file, temp_model_file):
        """Test that multiple predictions on same PDF give same result."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        
        result1 = pc.predict(dpi=7)
        result2 = pc.predict(dpi=7)
        
        assert result1['total_pages'] == result2['total_pages']
        assert result1['total_price'] == result2['total_price']
    
    def test_extraction_before_prediction(self, sample_pdf_file, temp_model_file):
        """Test that DataFrame is populated after prediction."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        
        assert pc.df is None or len(pc.df) == 0
        
        pc.predict(dpi=7)
        
        assert pc.df is not None
        assert len(pc.df) > 0
        assert 'price' in pc.df.columns
