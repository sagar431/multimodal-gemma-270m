#!/usr/bin/env python3
"""
Upload trained checkpoint to HuggingFace Hub.
This allows CI/CD to download the checkpoint for deployment.

Usage:
    python scripts/upload_checkpoint.py --username YOUR_HF_USERNAME
    
After running this, CI/CD can automatically download and deploy your model.
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi, upload_file, create_repo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def upload_checkpoint(username: str, checkpoint_path: str = None):
    """
    Upload checkpoint to HuggingFace Hub.
    
    Args:
        username: HuggingFace username
        checkpoint_path: Path to checkpoint file
    """
    api = HfApi()
    
    # Find checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt_path = Path(checkpoint_path)
    else:
        # Try to find checkpoint automatically
        possible_paths = [
            "models/checkpoints/gemma-270m-llava-training/final_model.ckpt",
            "models/checkpoints/final_model.ckpt",
            "models/checkpoints/last.ckpt",
        ]
        
        ckpt_path = None
        for path in possible_paths:
            if Path(path).exists():
                ckpt_path = Path(path)
                break
        
        if ckpt_path is None:
            log.error("❌ No checkpoint found! Please train the model first.")
            return
    
    log.info(f"📦 Found checkpoint: {ckpt_path}")
    log.info(f"   Size: {ckpt_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Create repo for checkpoints
    repo_id = f"{username}/multimodal-gemma-270m-checkpoints"
    
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False  # Set to True if you want private
        )
        log.info(f"✅ Repository ready: {repo_id}")
    except Exception as e:
        log.info(f"Repository note: {e}")
    
    # Upload checkpoint
    log.info(f"⬆️ Uploading checkpoint to {repo_id}...")
    
    upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo="final_model.ckpt",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload trained checkpoint"
    )
    
    log.info("=" * 60)
    log.info("✅ Checkpoint uploaded successfully!")
    log.info(f"   Repo: https://huggingface.co/{repo_id}")
    log.info("=" * 60)
    log.info("")
    log.info("📋 Next steps:")
    log.info("   1. Push your code to GitHub: git push origin main")
    log.info("   2. CI/CD will automatically download checkpoint and deploy!")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to HuggingFace Hub")
    parser.add_argument("--username", type=str, required=True, 
                        help="Your HuggingFace username")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (auto-detected if not provided)")
    
    args = parser.parse_args()
    
    upload_checkpoint(args.username, args.checkpoint)


if __name__ == "__main__":
    main()
