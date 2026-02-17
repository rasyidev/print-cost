"""
Unit tests for ModelManager service.

Tests model loading, caching, and error handling.
"""

import pytest
import pickle
from pathlib import Path

from src.services.model_manager import ModelManager
from src.exceptions import ModelLoadError


class TestModelLoading:
    """Test model loading functionality."""
    
    def test_load_model_from_file(self, temp_model_file):
        """Test loading a model from a pickle file."""
        model = ModelManager.load_model(temp_model_file)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_load_model_with_path_object(self, temp_model_file):
        """Test loading model using Path object."""
        model = ModelManager.load_model(Path(temp_model_file))
        
        assert model is not None
    
    def test_load_nonexistent_model_raises_error(self):
        """Test that loading nonexistent model raises ModelLoadError."""
        with pytest.raises(ModelLoadError) as exc_info:
            ModelManager.load_model("/nonexistent/path/model.pkl")
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_load_invalid_pickle_raises_error(self, tmp_path):
        """Test that loading invalid pickle file raises ModelLoadError."""
        invalid_file = tmp_path / "invalid.pkl"
        invalid_file.write_text("not a pickle file")
        
        with pytest.raises(ModelLoadError):
            ModelManager.load_model(str(invalid_file))


class TestModelCaching:
    """Test model caching functionality."""
    
    def setup_method(self):
        """Clear cache before each test."""
        ModelManager.clear_cache()
    
    def teardown_method(self):
        """Clear cache after each test."""
        ModelManager.clear_cache()
    
    def test_model_is_cached_after_first_load(self, temp_model_file):
        """Test that model is cached after first load."""
        # Load model
        model1 = ModelManager.load_model(temp_model_file, use_cache=True)
        
        # Check cache
        cache_info = ModelManager.get_cache_info()
        assert cache_info['cache_size'] == 1
        assert temp_model_file in cache_info['cached_models']
        
        # Load again - should be same object
        model2 = ModelManager.load_model(temp_model_file, use_cache=True)
        assert model1 is model2
    
    def test_cache_disabled_loads_new_instance(self, temp_model_file):
        """Test that disabling cache loads new instance each time."""
        model1 = ModelManager.load_model(temp_model_file, use_cache=False)
        model2 = ModelManager.load_model(temp_model_file, use_cache=False)
        
        # Should be different objects
        assert model1 is not model2
    
    def test_clear_cache_removes_all_models(self, temp_model_file, tmp_path, mock_model):
        """Test that clear_cache removes all cached models."""
        # Create and load multiple models
        model_file_2 = tmp_path / "model2.pkl"
        with open(model_file_2, "wb") as f:
            pickle.dump(mock_model, f)
        
        ModelManager.load_model(temp_model_file)
        ModelManager.load_model(str(model_file_2))
        
        assert ModelManager.get_cache_info()['cache_size'] == 2
        
        # Clear cache
        ModelManager.clear_cache()
        
        assert ModelManager.get_cache_info()['cache_size'] == 0
    
    def test_get_cache_info_structure(self, temp_model_file):
        """Test structure of cache info dictionary."""
        ModelManager.load_model(temp_model_file)
        
        cache_info = ModelManager.get_cache_info()
        
        assert 'cached_models' in cache_info
        assert 'cache_size' in cache_info
        assert isinstance(cache_info['cached_models'], list)
        assert isinstance(cache_info['cache_size'], int)


class TestDefaultModel:
    """Test default model loading."""
    
    def test_get_default_model_uses_config_path(self, monkeypatch, temp_model_file):
        """Test that get_default_model uses path from config."""
        # Patch the default model path in config
        import src.config as config
        monkeypatch.setattr(config, 'DEFAULT_MODEL_PATH', Path(temp_model_file))
        
        # Clear cache to ensure fresh load
        ModelManager.clear_cache()
        
        model = ModelManager.get_default_model()
        assert model is not None
    
    def test_get_default_model_raises_if_not_found(self, monkeypatch):
        """Test that get_default_model raises error if model not found."""
        # Patch to nonexistent path
        import src.config as config
        monkeypatch.setattr(config, 'DEFAULT_MODEL_PATH', Path("/nonexistent/model.pkl"))
        
        ModelManager.clear_cache()
        
        with pytest.raises(ModelLoadError):
            ModelManager.get_default_model()
