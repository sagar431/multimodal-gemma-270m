"""
PyTorch Lightning DataModule for LLaVA dataset
"""
import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from typing import Optional, Dict, Any
import logging

from .dataset import LLaVADataset, MultimodalCollator

logger = logging.getLogger(__name__)


class LLaVADataModule(L.LightningDataModule):
    """Lightning DataModule for LLaVA dataset"""
    
    def __init__(
        self,
        tokenizer,
        vision_processor,
        config: Dict[str, Any]
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.config = config
        
        # Data configuration
        data_config = config["data"]
        self.batch_size = config["training"]["batch_size"]
        self.num_workers = data_config.get("num_workers", 4)
        self.pin_memory = data_config.get("pin_memory", True)
        self.persistent_workers = data_config.get("persistent_workers", True)
        self.prefetch_factor = data_config.get("prefetch_factor", 2)  # Prefetch batches for speed
        
        # Dataset splits
        self.train_split = data_config.get("train_split", "train")
        self.val_split = data_config.get("val_split", "train")  # LLaVA doesn't have separate val
        self.val_size = data_config.get("val_size", 0.02)
        
        # Initialize datasets to None
        self.train_dataset = None
        self.val_dataset = None
        
        # Create collator
        self.collator = MultimodalCollator(
            tokenizer=self.tokenizer,
            vision_processor=self.vision_processor,
            config=self.config
        )
        
        logger.info("LLaVADataModule initialized")
    
    def prepare_data(self) -> None:
        """Download and prepare data (called only on rank 0)"""
        # This will download the dataset if not already cached
        try:
            LLaVADataset(
                config=self.config,
                split=self.train_split
            )
            logger.info("Dataset preparation completed")
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            raise
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training/validation/testing"""
        
        if stage == "fit" or stage is None:
            # Load full training dataset
            full_dataset = LLaVADataset(
                config=self.config,
                split=self.train_split
            )
            
            # Split into train and validation
            total_size = len(full_dataset)
            val_size = int(total_size * self.val_size)
            train_size = total_size - val_size
            
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            
            logger.info(f"Dataset split: {train_size} train, {val_size} validation")
            
        if stage == "test":
            # For testing, we'll use a small subset of the training data
            self.test_dataset = LLaVADataset(
                config=self.config,
                split=self.train_split
            )
            
        if stage == "predict":
            # For prediction, setup can be done dynamically
            pass
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=self.collator,
            drop_last=True  # Drop incomplete batches for consistent training
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=self.collator,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader"""
        if not hasattr(self, 'test_dataset') or self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            drop_last=False
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader"""
        # This can be implemented based on specific prediction needs
        return self.val_dataloader()
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after training/testing"""
        # Log dataset statistics if available
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            if hasattr(self.train_dataset.dataset, 'get_stats'):
                stats = self.train_dataset.dataset.get_stats()
                logger.info(f"Training dataset stats: {stats}")
        
        if hasattr(self, 'val_dataset') and self.val_dataset is not None:
            if hasattr(self.val_dataset.dataset, 'get_stats'):
                stats = self.val_dataset.dataset.get_stats()
                logger.info(f"Validation dataset stats: {stats}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded datasets"""
        info = {}
        
        if self.train_dataset is not None:
            info["train_size"] = len(self.train_dataset)
        
        if self.val_dataset is not None:
            info["val_size"] = len(self.val_dataset)
        
        info["batch_size"] = self.batch_size
        info["num_workers"] = self.num_workers
        
        return info
