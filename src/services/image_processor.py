"""
Image processing service for CMYK extraction and conversion.

This module handles all image-related operations including RGB to CMYK conversion
and CMYK percentage calculations for print cost estimation.
"""

from typing import Tuple
import numpy as np
from PIL import Image
import PIL.Image

from ..exceptions import ValidationError


class ImageProcessor:
    """
    Handle image processing operations for print cost calculation.
    
    This class provides methods to convert RGB images to CMYK color space
    and calculate the percentage distribution of each CMYK component.
    """
    
    SUPPORTED_IMAGE_FORMATS = ["jpg", "png", "jpeg", "webp", "heic"]
    CMYK_MIN = 0.0
    CMYK_MAX = 100.0
    EPSILON = 1e-10  # To avoid division by zero
    
    @staticmethod
    def rgb_to_cmyk(image: Image.Image | str) -> np.ndarray:
        """
        Convert an RGB image to CMYK color space.
        
        Args:
            image: PIL Image object or path to image file
            
        Returns:
            NumPy array with shape (height, width, 4) containing CMYK values
            
        Raises:
            ValidationError: If image type or format is invalid
        """
        # Validate input type
        if not isinstance(image, (str, PIL.Image.Image)):
            raise ValidationError(
                "image",
                f"Must be str or PIL.Image.Image, not {type(image).__name__}"
            )
        
        # Validate file extension if string path provided
        if isinstance(image, str):
            extension = image.split(".")[-1].lower()
            if extension not in ImageProcessor.SUPPORTED_IMAGE_FORMATS:
                raise ValidationError(
                    "image_format",
                    f"File type must be an image format. "
                    f"Supported: {', '.join(ImageProcessor.SUPPORTED_IMAGE_FORMATS)}"
                )
        
        # Load and convert image to RGB
        img = Image.open(image).convert("RGB") if isinstance(image, str) else image
        
        # Convert image to normalized array [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Convert RGB to CMY
        c = 1.0 - img_array[:, :, 0]
        m = 1.0 - img_array[:, :, 1]
        y = 1.0 - img_array[:, :, 2]
        
        # Calculate K (black) component
        k = np.minimum(np.minimum(c, m), y)
        
        # Adjust CMY values based on K to avoid division by zero
        c = (c - k) / (1.0 - k + ImageProcessor.EPSILON)
        m = (m - k) / (1.0 - k + ImageProcessor.EPSILON)
        y = (y - k) / (1.0 - k + ImageProcessor.EPSILON)
        
        # Stack into CMYK array
        return np.dstack((c, m, y, k))
    
    @staticmethod
    def calculate_cmyk_percentage(image: Image.Image | str) -> Tuple[float, float, float, float]:
        """
        Calculate the average percentage of each CMYK component in an image.
        
        This is the key feature extraction step for print cost prediction.
        
        Args:
            image: PIL Image object or path to image file
            
        Returns:
            Tuple of (cyan_%, magenta_%, yellow_%, black_%) rounded to 2 decimal places
            
        Raises:
            ValidationError: If image is invalid or CMYK values are out of range
        """
        # Convert to CMYK
        cmyk_array = ImageProcessor.rgb_to_cmyk(image)
        
        # Calculate mean percentage for each channel
        c_percent = float(np.mean(cmyk_array[:, :, 0]) * 100.0)
        m_percent = float(np.mean(cmyk_array[:, :, 1]) * 100.0)
        y_percent = float(np.mean(cmyk_array[:, :, 2]) * 100.0)
        k_percent = float(np.mean(cmyk_array[:, :, 3]) * 100.0)
        
        # Validate CMYK percentages are in valid range
        for name, value in [("cyan", c_percent), ("magenta", m_percent), 
                            ("yellow", y_percent), ("black", k_percent)]:
            if not (ImageProcessor.CMYK_MIN <= value <= ImageProcessor.CMYK_MAX):
                raise ValidationError(
                    f"cmyk_{name}",
                    f"CMYK {name} percentage {value:.2f} is outside valid range "
                    f"[{ImageProcessor.CMYK_MIN}, {ImageProcessor.CMYK_MAX}]"
                )
        
        return (
            round(c_percent, 2),
            round(m_percent, 2),
            round(y_percent, 2),
            round(k_percent, 2),
        )
