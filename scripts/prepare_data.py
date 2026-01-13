#!/usr/bin/env python3
"""
Script to prepare training data.
Downloads and processes the LLaVA dataset for training.
"""

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def prepare_llava_data(data_dir: str, subset_size: int = None):
    """
    Prepare LLaVA training data.
    
    Args:
        data_dir: Output directory for processed data
        subset_size: Optional limit on number of samples
    """
    from datasets import load_dataset
    
    log.info("Loading LLaVA dataset...")
    
    # Create output directory
    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load LLaVA conversations dataset
        dataset = load_dataset(
            "liuhaotian/LLaVA-Pretrain",
            split="train"
        )
        
        if subset_size:
            dataset = dataset.select(range(min(subset_size, len(dataset))))
        
        log.info(f"Dataset size: {len(dataset)}")
        
        # Save processed data
        output_path = output_dir / "llava_train.json"
        
        processed_data = []
        for i, sample in enumerate(dataset):
            processed_data.append({
                "id": i,
                "image": sample.get("image", ""),
                "conversations": sample.get("conversations", [])
            })
        
        with open(output_path, 'w') as f:
            json.dump(processed_data, f)
        
        log.info(f"✅ Data saved to: {output_path}")
        
    except Exception as e:
        log.warning(f"Could not load LLaVA dataset: {e}")
        log.info("Creating placeholder data for testing...")
        
        # Create placeholder data
        placeholder_data = [
            {
                "id": 0,
                "image": "",
                "conversations": [
                    {"from": "human", "value": "What is this image?"},
                    {"from": "assistant", "value": "This is a test image."}
                ]
            }
        ]
        
        output_path = output_dir / "llava_train.json"
        with open(output_path, 'w') as f:
            json.dump(placeholder_data, f)
        
        log.info(f"✅ Placeholder data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Limit dataset size for testing")
    
    args = parser.parse_args()
    
    log.info("=" * 60)
    log.info("📦 Data Preparation Script")
    log.info("=" * 60)
    
    prepare_llava_data(args.data_dir, args.subset_size)
    
    log.info("=" * 60)
    log.info("✅ Data preparation complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
