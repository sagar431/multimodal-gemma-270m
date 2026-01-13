#!/usr/bin/env python3
"""
Script to trace/export the trained Multimodal Gemma model for deployment.
Creates an optimized model that can be deployed to HuggingFace Spaces without 
the full training infrastructure.

Usage:
    python src/trace_model.py --ckpt_path <path_to_checkpoint> --output_path <output_model.pt>

Example:
    python src/trace_model.py --ckpt_path models/checkpoints/final_model.ckpt --output_path hf_space/model.pt
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, merge_configs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class MultimodalGemmaForDeployment(nn.Module):
    """
    Wrapper class for the Multimodal Gemma model optimized for deployment.
    This class simplifies the model interface for inference-only usage.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for inference"""
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True
            )
        return outputs


def load_model_from_checkpoint(ckpt_path: str, config: Dict[str, Any]) -> nn.Module:
    """
    Load model from Lightning checkpoint.
    
    Args:
        ckpt_path: Path to the checkpoint file
        config: Model configuration dictionary
    
    Returns:
        Loaded model in eval mode
    """
    log.info(f"Loading checkpoint from: {ckpt_path}")
    
    try:
        from src.models import MultimodalGemmaLightning
        
        # Load from checkpoint
        model = MultimodalGemmaLightning.load_from_checkpoint(
            ckpt_path,
            config=config,
            strict=False,
            map_location='cpu'
        )
        model.eval()
        
        log.info("✅ Model loaded successfully from checkpoint!")
        return model
        
    except Exception as e:
        log.error(f"Error loading checkpoint: {e}")
        raise


def create_dummy_model_for_demo(config: Dict[str, Any]) -> nn.Module:
    """
    Create a dummy model for demonstration purposes when no checkpoint is available.
    This is useful for CI/CD testing.
    """
    log.info("Creating dummy model for demonstration...")
    
    try:
        from src.models import MultimodalGemmaLightning
        
        # Create model with default config
        model = MultimodalGemmaLightning(config)
        model.eval()
        
        log.info("✅ Dummy model created successfully!")
        return model
        
    except Exception as e:
        log.warning(f"Could not create full model: {e}")
        log.info("Creating minimal placeholder model instead...")
        
        # Create a minimal placeholder
        class PlaceholderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(768, 768)
            
            def forward(self, x):
                return self.linear(x)
        
        return PlaceholderModel()


def export_model_for_deployment(model, config: Dict[str, Any], output_path: str):
    """
    Export the model in a format suitable for HuggingFace Spaces deployment.
    For multimodal models, we save the state dict and config rather than tracing.
    
    Args:
        model: The PyTorch Lightning model
        config: Model configuration
        output_path: Path to save the exported model
    """
    log.info(f"Exporting model for deployment to: {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # For complex multimodal models, save state dict + config
    export_dict = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_type': 'multimodal_gemma',
        'version': '1.0'
    }
    
    torch.save(export_dict, output_path)
    
    # Print file size
    file_size = output_path.stat().st_size / (1024 * 1024)
    log.info(f"✅ Model exported successfully!")
    log.info(f"   Path: {output_path}")
    log.info(f"   Size: {file_size:.2f} MB")
    
    return output_path


def save_config_for_deployment(config: Dict[str, Any], output_dir: str):
    """Save configuration file for deployment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)
    log.info(f"✅ Config saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Multimodal Gemma model for deployment")
    parser.add_argument("--ckpt_path", type=str, default=None, 
                        help="Path to Lightning checkpoint file")
    parser.add_argument("--output_path", type=str, default="hf_space/model.pt", 
                        help="Output path for exported model")
    parser.add_argument("--config_path", type=str, default="configs/model_config.yaml",
                        help="Path to model config file")
    parser.add_argument("--use_dummy", action="store_true",
                        help="Create a dummy model for testing (no checkpoint needed)")

    args = parser.parse_args()

    log.info("=" * 60)
    log.info("🔧 Multimodal Gemma Model Export Script")
    log.info("=" * 60)
    log.info(f"Checkpoint: {args.ckpt_path if args.ckpt_path else 'None (using dummy)'}")
    log.info(f"Output: {args.output_path}")
    log.info(f"Config: {args.config_path}")
    log.info("=" * 60)

    # Load configuration
    try:
        model_config = load_config("configs/model_config.yaml")
        training_config = load_config("configs/training_config.yaml")
        data_config = load_config("configs/data_config.yaml")
        config = merge_configs([model_config, training_config, data_config])
        log.info("✅ Configuration loaded successfully!")
    except Exception as e:
        log.warning(f"Could not load full config: {e}")
        config = {
            "model": {
                "gemma_model_name": "google/gemma-2b",
                "vision_model_name": "openai/clip-vit-large-patch14"
            }
        }

    # Load or create model
    if args.ckpt_path and Path(args.ckpt_path).exists():
        model = load_model_from_checkpoint(args.ckpt_path, config)
    elif args.use_dummy:
        model = create_dummy_model_for_demo(config)
    else:
        # Try to find checkpoint automatically
        checkpoint_paths = [
            "models/checkpoints/gemma-270m-llava-training/final_model.ckpt",
            "models/checkpoints/final_model.ckpt",
            "models/checkpoints/last.ckpt"
        ]
        
        ckpt_found = None
        for ckpt_path in checkpoint_paths:
            if Path(ckpt_path).exists():
                ckpt_found = ckpt_path
                break
        
        if ckpt_found:
            log.info(f"Found checkpoint at: {ckpt_found}")
            model = load_model_from_checkpoint(ckpt_found, config)
        else:
            log.warning("No checkpoint found, creating dummy model for CI/CD demo...")
            model = create_dummy_model_for_demo(config)

    # Export model
    export_model_for_deployment(model, config, args.output_path)
    
    # Save config for deployment
    output_dir = Path(args.output_path).parent
    save_config_for_deployment(config, output_dir)
    
    log.info("=" * 60)
    log.info("✅ Model export complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
