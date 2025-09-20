"""
Configuration utilities
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        logger.info(f"Loaded configuration from: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries"""
    merged = {}
    
    for config in configs:
        merged.update(config)
    
    logger.info(f"Merged {len(configs)} configuration files")
    return merged


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to: {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {save_path}: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure"""
    required_sections = ["model", "training", "data"]
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate model config
    model_config = config["model"]
    required_model_keys = ["gemma_model_name", "vision_model_name", "lora"]
    for key in required_model_keys:
        if key not in model_config:
            logger.error(f"Missing required model config key: {key}")
            return False
    
    # Validate training config
    training_config = config["training"]
    required_training_keys = ["max_epochs", "batch_size", "lora_lr", "projector_lr"]
    for key in required_training_keys:
        if key not in training_config:
            logger.error(f"Missing required training config key: {key}")
            return False
    
    # Validate data config
    data_config = config["data"]
    required_data_keys = ["dataset_name", "max_length", "image_size"]
    for key in required_data_keys:
        if key not in data_config:
            logger.error(f"Missing required data config key: {key}")
            return False
    
    logger.info("Configuration validation passed")
    return True


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values"""
    def deep_update(base_dict, update_dict):
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    import copy
    updated_config = copy.deepcopy(config)
    deep_update(updated_config, updates)
    
    logger.info("Configuration updated")
    return updated_config
