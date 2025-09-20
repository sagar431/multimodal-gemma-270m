"""
Training utilities
"""
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingUtils:
    """Utility functions for training"""
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": frozen_params,
            "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0
        }
    
    @staticmethod
    def print_model_summary(model: torch.nn.Module, model_name: str = "Model") -> None:
        """Print detailed model summary"""
        params = TrainingUtils.count_parameters(model)
        
        logger.info(f"\n{model_name} Summary:")
        logger.info(f"  Total parameters: {params['total']:,}")
        logger.info(f"  Trainable parameters: {params['trainable']:,}")
        logger.info(f"  Frozen parameters: {params['frozen']:,}")
        logger.info(f"  Trainable percentage: {params['trainable_percentage']:.2f}%")
    
    @staticmethod
    def save_model_state(
        model: torch.nn.Module, 
        path: str, 
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model state with additional information"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
        }
        
        if additional_info:
            state_dict.update(additional_info)
        
        torch.save(state_dict, save_path)
        logger.info(f"Model state saved to: {save_path}")
    
    @staticmethod
    def load_model_state(model: torch.nn.Module, path: str, strict: bool = True) -> Dict[str, Any]:
        """Load model state and return additional information"""
        checkpoint = torch.load(path, map_location="cpu")
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            logger.info(f"Model state loaded from: {path}")
            
            # Return additional info
            additional_info = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
            return additional_info
        else:
            # Assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint, strict=strict)
            logger.info(f"Model state loaded from: {path}")
            return {}
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get information about available devices"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info["cuda_current_device"] = torch.cuda.current_device()
            info["cuda_device_name"] = torch.cuda.get_device_name()
            info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return info
    
    @staticmethod
    def log_device_info() -> None:
        """Log device information"""
        info = TrainingUtils.get_device_info()
        
        logger.info("\nDevice Information:")
        logger.info(f"  CUDA Available: {info['cuda_available']}")
        
        if info['cuda_available']:
            logger.info(f"  CUDA Device Count: {info['cuda_device_count']}")
            logger.info(f"  Current Device: {info['cuda_current_device']}")
            logger.info(f"  Device Name: {info['cuda_device_name']}")
            logger.info(f"  Total Memory: {info['cuda_memory_total']:.2f} GB")
        else:
            logger.info("  Using CPU for training")
    
    @staticmethod
    def cleanup_memory() -> None:
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
