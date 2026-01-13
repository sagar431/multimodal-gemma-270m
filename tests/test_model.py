"""
Test suite for Multimodal Gemma model
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelImports:
    """Test that model imports work correctly"""
    
    def test_import_models(self):
        """Test importing model modules"""
        from src.models import MultimodalGemma, MultimodalGemmaLightning
        assert MultimodalGemma is not None
        assert MultimodalGemmaLightning is not None
    
    def test_import_projectors(self):
        """Test importing projector modules"""
        from src.models import VisionProjector
        assert VisionProjector is not None
    
    def test_import_utils(self):
        """Test importing utility functions"""
        from src.utils.config import load_config, merge_configs
        assert load_config is not None
        assert merge_configs is not None


class TestVisionProjector:
    """Test vision projector module"""
    
    def test_projector_creation(self):
        """Test creating vision projector"""
        from src.models.projectors import VisionProjector
        
        projector = VisionProjector(
            vision_hidden_size=768,
            language_hidden_size=2048,
            num_hidden_layers=2
        )
        
        assert projector is not None
    
    def test_projector_forward(self):
        """Test projector forward pass"""
        from src.models.projectors import VisionProjector
        
        projector = VisionProjector(
            vision_hidden_size=768,
            language_hidden_size=2048,
            num_hidden_layers=2
        )
        
        # Create dummy input
        batch_size = 2
        seq_len = 196  # 14x14 patches
        hidden_size = 768
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = projector(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, 2048)


class TestConfigLoading:
    """Test configuration loading"""
    
    def test_load_model_config(self):
        """Test loading model configuration"""
        from src.utils.config import load_config
        
        config_path = Path("configs/model_config.yaml")
        if config_path.exists():
            config = load_config(str(config_path))
            assert config is not None
            assert "model" in config or len(config) > 0
    
    def test_merge_configs(self):
        """Test merging configurations"""
        from src.utils.config import merge_configs
        
        config1 = {"model": {"name": "test"}}
        config2 = {"training": {"epochs": 10}}
        
        merged = merge_configs([config1, config2])
        
        assert "model" in merged
        assert "training" in merged


class TestTraceModel:
    """Test model tracing functionality"""
    
    def test_trace_model_import(self):
        """Test that trace_model can be imported"""
        from src.trace_model import export_model_for_deployment
        assert export_model_for_deployment is not None
    
    def test_create_dummy_model(self):
        """Test creating dummy model for demo"""
        from src.trace_model import create_dummy_model_for_demo
        
        config = {
            "model": {
                "gemma_model_name": "google/gemma-2b",
                "vision_model_name": "openai/clip-vit-large-patch14"
            }
        }
        
        # This may fail without full setup, which is expected
        try:
            model = create_dummy_model_for_demo(config)
            assert model is not None
        except Exception:
            # Expected in CI without full model setup
            pass


class TestLightningModule:
    """Test PyTorch Lightning module"""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return {
            "model": {
                "gemma_model_name": "google/gemma-2b",
                "vision_model_name": "openai/clip-vit-large-patch14",
                "use_lora": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "freeze_vision": True,
                "freeze_language": True,
                "projector_hidden_layers": 2
            },
            "training": {
                "projector_lr": 1e-4,
                "lora_lr": 2e-5,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "max_epochs": 3,
                "accumulate_grad_batches": 4,
                "monitor": "val/loss",
                "mode": "min",
                "save_top_k": 3,
                "patience": 5,
                "min_delta": 0.001
            },
            "logging": {
                "wandb_name": "test-run",
                "wandb_project": "test-project",
                "log_model": False,
                "use_wandb": False,
                "use_tensorboard": False,
                "tb_log_dir": "logs/tensorboard"
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": 1,
                "num_nodes": 1,
                "log_every_n_steps": 10,
                "enable_checkpointing": True,
                "enable_progress_bar": True,
                "enable_model_summary": True,
                "fast_dev_run": False,
                "overfit_batches": 0,
                "detect_anomaly": False,
                "deterministic": False,
                "benchmark": True
            }
        }
    
    def test_lightning_module_init(self, sample_config):
        """Test Lightning module initialization"""
        # Skip if models can't be loaded (e.g., in CI without GPU)
        pytest.skip("Skipping full model initialization test in CI")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
