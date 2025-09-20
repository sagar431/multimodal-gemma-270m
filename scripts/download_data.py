#!/usr/bin/env python3
"""
Script to download and prepare the LLaVA dataset
"""
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def download_llava_dataset(cache_dir: str = "./data/cache"):
    """Download the LLaVA instruction dataset"""
    setup_logging()
    
    logger.info("Starting LLaVA dataset download...")
    
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the dataset
        dataset = load_dataset(
            "liuhaotian/LLaVA-Instruct-150K",
            cache_dir=cache_dir,
            verification_mode="no_checks"
        )
        
        logger.info(f"Successfully downloaded LLaVA dataset")
        logger.info(f"Dataset size: {len(dataset['train'])} samples")
        logger.info(f"Cache location: {cache_dir}")
        
        # Print sample data
        sample = dataset['train'][0]
        logger.info(f"Sample data keys: {list(sample.keys())}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download LLaVA dataset")
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default="./data/cache",
        help="Directory to cache the dataset"
    )
    
    args = parser.parse_args()
    
    download_llava_dataset(args.cache_dir)
