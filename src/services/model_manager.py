"""
Model management service for loading and caching ML models.

This module handles model loading, versioning, and provides a centralized
interface for model inference operations.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional
import logging

from ..exceptions import ModelLoadError
from .. import config

logger = logging.getLogger(__name__)

#: First bytes of a Git-LFS pointer file. The real XGBoost artifact is ~400 KB
#: of pickle; a clone without `git lfs pull` holds one of these instead, and
#: unpickling it would fail with a confusing error much later.
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


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
    def verify_model_artifact(
        cls,
        model_path: Path | str,
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
    ) -> dict[str, Any]:
        """
        Verify that a model artifact is a real pickled model, not an LFS pointer.

        Args:
            model_path: Path to the model artifact
            expected_sha256: Optional pinned SHA-256 of the artifact
            expected_size_bytes: Optional pinned size in bytes

        Returns:
            Dict with path, size_bytes and sha256

        Raises:
            ModelLoadError: If the file is missing, is a Git-LFS pointer, or does
                not match the expected size/hash.
        """
        path = Path(model_path)

        if not path.exists():
            raise ModelLoadError(
                str(model_path),
                FileNotFoundError(f"Model file not found: {model_path}")
            )

        size_bytes = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(len(LFS_POINTER_PREFIX))

        if head == LFS_POINTER_PREFIX:
            raise ModelLoadError(
                str(model_path),
                ValueError(
                    "Artifact is a Git-LFS pointer, not the model. Run "
                    "`git lfs pull` or `python scripts/fetch_model.py` first."
                )
            )

        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        sha256 = digest.hexdigest()

        if expected_size_bytes is not None and size_bytes != expected_size_bytes:
            raise ModelLoadError(
                str(model_path),
                ValueError(
                    f"Size mismatch: expected {expected_size_bytes} bytes, "
                    f"found {size_bytes}"
                )
            )

        if expected_sha256 is not None and sha256 != expected_sha256:
            raise ModelLoadError(
                str(model_path),
                ValueError(
                    f"SHA-256 mismatch: expected {expected_sha256}, found {sha256}"
                )
            )

        return {"path": str(path), "size_bytes": size_bytes, "sha256": sha256}

    @classmethod
    def get_default_model(cls) -> Any:
        """
        Load the default model specified in configuration.

        The path is read from `config` at call time (not bound at import time)
        so it stays monkeypatchable, and the artifact is verified against the
        pinned SHA-256 before loading so a Git-LFS pointer can never be mistaken
        for a model.

        Returns:
            Default model object

        Raises:
            ModelLoadError: If default model cannot be loaded
        """
        model_path = Path(config.DEFAULT_MODEL_PATH)

        # Only the pinned artifact is hash-checked; a path injected by tests or
        # by a custom deployment is still checked for the LFS-pointer trap.
        expected_sha256 = None
        expected_size_bytes = None
        if model_path.name == config.DEFAULT_MODEL_FILE:
            expected_sha256 = config.DEFAULT_MODEL_SHA256
            expected_size_bytes = config.DEFAULT_MODEL_SIZE_BYTES

        cls.verify_model_artifact(model_path, expected_sha256, expected_size_bytes)
        return cls.load_model(model_path)
