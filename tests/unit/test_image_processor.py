"""
Unit tests for ImageProcessor service.

Tests RGB to CMYK conversion and CMYK percentage calculation.
"""

import pytest
import numpy as np
from PIL import Image

from src.services.image_processor import ImageProcessor
from src.exceptions import ValidationError


class TestRGBtoCMYK:
    """Test RGB to CMYK conversion."""
    
    def test_convert_pil_image(self, sample_rgb_image):
        """Test conversion of PIL Image object."""
        result = ImageProcessor.rgb_to_cmyk(sample_rgb_image)
        
        assert result.shape == (100, 100, 4)
        assert result.dtype == np.float32 or result.dtype == np.float64
        
        # Check CMYK values are in valid range [0, 1]
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_pure_white_image(self, white_image):
        """Test that pure white converts to all zeros in CMYK."""
        result = ImageProcessor.rgb_to_cmyk(white_image)
        
        # White should have minimal CMYK values
        assert np.mean(result) < 0.1
    
    def test_pure_black_image(self, black_image):
        """Test that pure black converts to maximum K value."""
        result = ImageProcessor.rgb_to_cmyk(black_image)
        
        # Black should have high K value (channel 3)
        k_channel = result[:, :, 3]
        assert np.mean(k_channel) > 0.9
    
    def test_grayscale_image(self, grayscale_image):
        """Test grayscale image conversion."""
        result = ImageProcessor.rgb_to_cmyk(grayscale_image)
        
        # Grayscale should have equal CMY and some K
        c_mean = np.mean(result[:, :, 0])
        m_mean = np.mean(result[:, :, 1])
        y_mean = np.mean(result[:, :, 2])
        
        assert np.isclose(c_mean, m_mean, atol=0.01)
        assert np.isclose(m_mean, y_mean, atol=0.01)
    
    def test_invalid_input_type(self):
        """Test that invalid input type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ImageProcessor.rgb_to_cmyk(123)
        
        assert "image" in str(exc_info.value).lower()


class TestCalculateCMYKPercentage:
    """Test CMYK percentage calculation."""
    
    def test_calculate_percentage_returns_tuple(self, sample_rgb_image):
        """Test that calculation returns a 4-tuple of floats."""
        result = ImageProcessor.calculate_cmyk_percentage(sample_rgb_image)
        
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(x, float) for x in result)
    
    def test_percentage_values_in_valid_range(self, sample_rgb_image):
        """Test that percentage values are between 0 and 100."""
        c, m, y, k = ImageProcessor.calculate_cmyk_percentage(sample_rgb_image)
        
        assert 0 <= c <= 100
        assert 0 <= m <= 100
        assert 0 <= y <= 100
        assert 0 <= k <= 100
    
    def test_white_image_low_percentages(self, white_image):
        """Test that white image has low CMYK percentages."""
        c, m, y, k = ImageProcessor.calculate_cmyk_percentage(white_image)
        
        assert c < 10
        assert m < 10
        assert y < 10
        assert k < 10
    
    def test_black_image_high_k_percentage(self, black_image):
        """Test that black image has high K percentage."""
        c, m, y, k = ImageProcessor.calculate_cmyk_percentage(black_image)
        
        assert k > 90
    
    def test_values_rounded_to_two_decimals(self, sample_rgb_image):
        """Test that values are rounded to 2 decimal places."""
        c, m, y, k = ImageProcessor.calculate_cmyk_percentage(sample_rgb_image)
        
        # Check that values have at most 2 decimal places
        assert c == round(c, 2)
        assert m == round(m, 2)
        assert y == round(y, 2)
        assert k == round(k, 2)
    
    def test_consistent_results(self, sample_rgb_image):
        """Test that multiple calls with same image return same result."""
        result1 = ImageProcessor.calculate_cmyk_percentage(sample_rgb_image)
        result2 = ImageProcessor.calculate_cmyk_percentage(sample_rgb_image)
        
        assert result1 == result2


class TestImageProcessorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_small_image(self):
        """Test with a very small 1x1 pixel image."""
        img_array = np.array([[[255, 0, 0]]], dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        c, m, y, k = ImageProcessor.calculate_cmyk_percentage(img)
        
        # Red color should have low C, high M and Y
        assert c < 10
        assert m > 80
        assert y > 80
    
    def test_very_large_values_dont_exceed_range(self):
        """Test that even with edge case colors, values stay in range."""
        # Create image with extreme reds
        img_array = np.full((50, 50, 3), [255, 0, 0], dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        c, m, y, k = ImageProcessor.calculate_cmyk_percentage(img)
        
        assert 0 <= c <= 100
        assert 0 <= m <= 100
        assert 0 <= y <= 100
        assert 0 <= k <= 100
