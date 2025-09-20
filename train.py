#!/usr/bin/env python3
"""
Main training script for Multimodal Gemma with PyTorch Lightning
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    RichProgressBar
)
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

# Import our modules
from src.models import MultimodalGemmaLightning
from src.data import LLaVADataModule
from src.utils.config import load_config, merge_configs
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def setup_callbacks(config: Dict[str, Any]) -> list:
    """Setup Lightning callbacks"""
    callbacks = []
    
    # Rich Progress Bar (as requested)
    callbacks.append(RichProgressBar())
    
    # Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"models/checkpoints/{config['logging']['wandb_name']}",
        filename="multimodal-gemma-{epoch:02d}-{val/loss:.2f}",
        monitor=config["training"]["monitor"],
        mode=config["training"]["mode"],
        save_top_k=config["training"]["save_top_k"],
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early Stopping
    early_stopping = EarlyStopping(
        monitor=config["training"]["monitor"],
        patience=config["training"]["patience"],
        mode=config["training"]["mode"],
        min_delta=config["training"]["min_delta"],
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_loggers(config: Dict[str, Any]) -> list:
    """Setup Lightning loggers"""
    loggers = []
    
    # Weights & Biases Logger
    if config["logging"].get("use_wandb", False):
        wandb_logger = WandbLogger(
            project=config["logging"]["wandb_project"],
            name=config["logging"]["wandb_name"],
            log_model=config["logging"]["log_model"],
            save_dir="logs/wandb"
        )
        loggers.append(wandb_logger)
    
    # TensorBoard Logger
    if config["logging"].get("use_tensorboard", False):
        tb_logger = TensorBoardLogger(
            save_dir=config["logging"]["tb_log_dir"],
            name=config["logging"]["wandb_name"]
        )
        loggers.append(tb_logger)
    
    return loggers


def main(config: Dict[str, Any]):
    """Main training function"""
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger.info("Starting Multimodal Gemma training")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Set seeds for reproducibility
    L.seed_everything(42)
    
    # Create directories
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize model
    logger.info("Initializing model...")
    model = MultimodalGemmaLightning(config)
    
    # Initialize data module
    logger.info("Initializing data module...")
    datamodule = LLaVADataModule(
        tokenizer=model.model.tokenizer,
        vision_processor=model.model.vision_processor,
        config=config
    )
    
    # Setup callbacks and loggers
    callbacks = setup_callbacks(config)
    loggers = setup_loggers(config)
    
    # Initialize trainer
    trainer_config = config["trainer"]
    training_config = config["training"]
    
    trainer = Trainer(
        # Hardware
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        num_nodes=trainer_config["num_nodes"],
        
        # Training
        max_epochs=training_config["max_epochs"],
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        gradient_clip_val=training_config["gradient_clip_val"],
        precision=training_config["precision"],
        strategy=training_config["strategy"],
        
        # Validation
        val_check_interval=training_config["val_check_interval"],
        limit_val_batches=training_config["limit_val_batches"],
        
        # Logging and monitoring
        log_every_n_steps=trainer_config["log_every_n_steps"],
        enable_checkpointing=trainer_config["enable_checkpointing"],
        enable_progress_bar=trainer_config["enable_progress_bar"],
        enable_model_summary=trainer_config["enable_model_summary"],
        
        # Callbacks and loggers
        callbacks=callbacks,
        logger=loggers,
        
        # Debugging
        fast_dev_run=trainer_config["fast_dev_run"],
        overfit_batches=trainer_config["overfit_batches"],
        detect_anomaly=trainer_config["detect_anomaly"],
        
        # Performance
        deterministic=trainer_config["deterministic"],
        benchmark=trainer_config["benchmark"],
    )
    
    # Log model architecture
    logger.info("Model architecture:")
    logger.info(f"Language model: {config['model']['gemma_model_name']}")
    logger.info(f"Vision model: {config['model']['vision_model_name']}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.fit(model, datamodule)
        logger.info("Training completed successfully!")
        
        # Save final model
        final_checkpoint_path = f"models/checkpoints/{config['logging']['wandb_name']}/final_model.ckpt"
        trainer.save_checkpoint(final_checkpoint_path)
        logger.info(f"Final model saved to: {final_checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Print dataset statistics
    if hasattr(datamodule, 'get_dataset_info'):
        dataset_info = datamodule.get_dataset_info()
        logger.info(f"Dataset info: {dataset_info}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point"""
    # Convert OmegaConf to regular dict for easier handling
    config = OmegaConf.to_container(cfg, resolve=True)
    main(config)


def direct_main():
    """Direct entry point without Hydra (for manual config loading)"""
    # Load configurations manually
    model_config = load_config("configs/model_config.yaml")
    training_config = load_config("configs/training_config.yaml")
    data_config = load_config("configs/data_config.yaml")
    
    # Merge configurations
    config = merge_configs([model_config, training_config, data_config])
    
    main(config)


if __name__ == "__main__":
    # Check if we want to use Hydra or direct config loading
    if "--config-path" in " ".join(os.sys.argv) or len(os.sys.argv) == 1:
        # Use Hydra
        hydra_main()
    else:
        # Use direct config loading
        direct_main()
