"""
Custom Lightning callbacks
"""
import lightning as L
from lightning.pytorch.callbacks import Callback
import torch
from typing import Any
import logging

logger = logging.getLogger(__name__)


class CustomCallback(Callback):
    """Custom callback for monitoring training progress"""
    
    def __init__(self):
        super().__init__()
        self.start_time = None
    
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called when training starts"""
        import time
        self.start_time = time.time()
        logger.info("Training started")
        
        # Log model info
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called when training ends"""
        if self.start_time:
            import time
            duration = time.time() - self.start_time
            logger.info(f"Training completed in {duration:.2f} seconds")
    
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called at the start of each training epoch"""
        logger.info(f"Starting epoch {trainer.current_epoch + 1}/{trainer.max_epochs}")
    
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called at the end of validation epoch"""
        if trainer.logged_metrics:
            val_loss = trainer.logged_metrics.get("val/loss", None)
            if val_loss is not None:
                logger.info(f"Validation loss: {val_loss:.4f}")


class MemoryMonitorCallback(Callback):
    """Monitor GPU memory usage during training"""
    
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_train_batch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Log memory usage"""
        if batch_idx % self.log_every_n_steps == 0 and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            pl_module.log("train/memory_allocated_gb", memory_allocated, on_step=True)
            pl_module.log("train/memory_reserved_gb", memory_reserved, on_step=True)
