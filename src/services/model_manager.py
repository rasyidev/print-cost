"""
Model management service for loading and caching ML models.

This module handles model loading, versioning, and provides a centralized
interface for model inference operations.
"""

import pickle
from pathlib import Path
from typing import Any, Optional
import logging

from ..exceptions import ModelLoadError
from ..config import DEFAULT_MODEL_PATH, MODEL_VERSION

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manage machine learning model loading and caching.
    
    This class provides a singleton-like model cache to avoid reloading
    models on every prediction request.
    """
    
    _model_cache: dict[str, Any] = {}
    
    @classmethod
    def load_model(cls, model_path: Path | str, use_cache: bool = True) -> Any:
        """
        Load a machine learning model from disk.
        
        Args:
            model_path: Path to the pickled model file
            use_cache: Whether to use cached model if available
            
        Returns:
            Loaded model object
            
        Raises:
            ModelLoadError: If model file cannot be loaded
        """
        model_path_str = str(model_path)
        
        # Return cached model if available and caching is enabled
        if use_cache and model_path_str in cls._model_cache:
            logger.info(f"Loading model from cache: {model_path_str}")
            return cls._model_cache[model_path_str]
        
        # Validate model file exists
        if not Path(model_path).exists():
            raise ModelLoadError(
                model_path_str,
                FileNotFoundError(f"Model file not found: {model_path_str}")
            )
        
        # Load model
        try:
            logger.info(f"Loading model from disk: {model_path_str}")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            # Cache the model
            if use_cache:
                cls._model_cache[model_path_str] = model
                logger.info(f"Model cached successfully: {model_path_str}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path_str}: {str(e)}")
            raise ModelLoadError(model_path_str, e)
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached models from memory."""
        cls._model_cache.clear()
        logger.info("Model cache cleared")
    
    @classmethod
    def get_cache_info(cls) -> dict[str, Any]:
        """
        Get information about cached models.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_models": list(cls._model_cache.keys()),
            "cache_size": len(cls._model_cache),
        }
    
    @classmethod
    def get_default_model(cls) -> Any:
        """
        Load the default model specified in configuration.
        
        Returns:
            Default model object
            
        Raises:
            ModelLoadError: If default model cannot be loaded
        """
        return cls.load_model(DEFAULT_MODEL_PATH)
