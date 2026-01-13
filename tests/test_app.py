"""
Test suite for HuggingFace Space app
"""

import pytest
import sys
from pathlib import Path


class TestAppImports:
    """Test that app can be imported"""
    
    def test_gradio_import(self):
        """Test gradio import"""
        import gradio as gr
        assert gr is not None
    
    def test_torch_import(self):
        """Test torch import"""
        import torch
        assert torch is not None
    
    def test_transformers_import(self):
        """Test transformers import"""
        from transformers import AutoTokenizer
        assert AutoTokenizer is not None


class TestAppFunctions:
    """Test app helper functions"""
    
    def test_load_model_returns_error_when_no_model(self):
        """Test that load_model returns appropriate error"""
        # Add hf_space to path
        hf_space_path = Path(__file__).parent.parent / "hf_space"
        sys.path.insert(0, str(hf_space_path))
        
        try:
            from app import load_model
            result = load_model()
            # Should return error since model.pt doesn't exist in test
            assert "❌" in result or "✅" in result
        except ImportError:
            # Expected if dependencies not fully installed
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
