"""
Pytest fixtures for the Print Cost test suite.

This module provides reusable fixtures for testing including mock data,
sample images, and test PDF documents.
"""

import pytest
import numpy as np
from PIL import Image
import io
import pymupdf
from pathlib import Path


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    # Create a 100x100 RGB image with a gradient
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Add some color variety
    img_array[:50, :50] = [255, 0, 0]    # Red
    img_array[:50, 50:] = [0, 255, 0]     # Green
    img_array[50:, :50] = [0, 0, 255]     # Blue
    img_array[50:, 50:] = [128, 128, 128] # Gray
    
    return Image.fromarray(img_array, 'RGB')


@pytest.fixture
def grayscale_image():
    """Create a grayscale image for testing."""
    img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
    return Image.fromarray(img_array, 'RGB')


@pytest.fixture
def black_image():
    """Create a pure black image for testing edge cases."""
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    return Image.fromarray(img_array, 'RGB')


@pytest.fixture
def white_image():
    """Create a pure white image for testing edge cases."""
    img_array = np.full((100, 100, 3), 255, dtype=np.uint8)
    return Image.fromarray(img_array, 'RGB')


@pytest.fixture
def sample_cmyk_data():
    """Provide sample CMYK percentage data."""
    return [
        {"c": 10.5, "m": 15.2, "y": 20.3, "k": 5.1},   # Light color
        {"c": 50.0, "m": 60.0, "y": 70.0, "k": 10.0},  # Medium color
        {"c": 80.0, "m": 85.0, "y": 90.0, "k": 20.0},  # Heavy color
        {"c": 5.0, "m": 5.0, "y": 5.0, "k": 90.0},     # Mostly black
    ]


@pytest.fixture
def mock_model():
    """Create a mock ML model for testing."""
    class MockModel:
        def predict(self, X):
            """Mock prediction that returns label 0 for all inputs."""
            return np.zeros(len(X), dtype=int)
    
    return MockModel()


@pytest.fixture
def mock_model_varied():
    """Create a mock ML model with varied predictions."""
    class MockModelVaried:
        def predict(self, X):
            """Mock prediction with varied labels based on input."""
            predictions = []
            for _, row in X.iterrows():
                cmyk_sum = row.get('cmyk', row.sum())
                if cmyk_sum < 50:
                    predictions.append(0)  # Mono
                elif cmyk_sum < 150:
                    predictions.append(1)  # Color Light
                elif cmyk_sum < 250:
                    predictions.append(2)  # Color Standard
                elif cmyk_sum < 300:
                    predictions.append(3)  # Color Heavy
                else:
                    predictions.append(4)  # Full Color
            return np.array(predictions)
    
    return MockModelVaried()


@pytest.fixture
def temp_model_file(tmp_path, mock_model):
    """Create a temporary model file for testing."""
    import pickle
    model_path = tmp_path / "test_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(mock_model, f)
    return str(model_path)


@pytest.fixture
def sample_pdf_bytes():
    """Create a simple in-memory PDF for testing."""
    # Create a simple PDF with 3 pages
    pdf = pymupdf.open()
    
    # Page 1: Red page
    page1 = pdf.new_page(width=595, height=842)  # A4 size
    page1.insert_text((100, 100), "Test Page 1", fontsize=20)
    shape = page1.new_shape()
    shape.draw_rect(pymupdf.Rect(50, 150, 545, 200))
    shape.finish(fill=(1, 0, 0))  # Red fill
    shape.commit()
    
    # Page 2: Green page  
    page2 = pdf.new_page(width=595, height=842)
    page2.insert_text((100, 100), "Test Page 2", fontsize=20)
    shape = page2.new_shape()
    shape.draw_rect(pymupdf.Rect(50, 150, 545, 200))
    shape.finish(fill=(0, 1, 0))  # Green fill
    shape.commit()
    
    # Page 3: Black text only (mono)
    page3 = pdf.new_page(width=595, height=842)
    page3.insert_text((100, 100), "Test Page 3 - Monochrome", fontsize=20)
    
    # Save to bytes
    pdf_bytes = pdf.tobytes()
    pdf.close()
    
    return pdf_bytes


@pytest.fixture
def sample_pdf_file(tmp_path, sample_pdf_bytes):
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test_document.pdf"
    with open(pdf_path, "wb") as f:
        f.write(sample_pdf_bytes)
    return str(pdf_path)


@pytest.fixture
def expected_price_categories():
    """Provide expected price categories for assertion."""
    return {
        500: "Mono Print",
        750: "Color Light",
        1000: "Color Standard",
        1500: "Color Heavy",
        2000: "Full Color – Dark & Mixed",
    }


@pytest.fixture
def sample_result_dict():
    """Provide a sample result dictionary structure."""
    return {
        "total_pages": 10,
        "total_price": 7500,
        "details": [
            {"price": 500, "pages": 5, "subtotal": 2500, "category": "Mono Print"},
            {"price": 1000, "pages": 5, "subtotal": 5000, "category": "Color Standard"},
        ],
    }
