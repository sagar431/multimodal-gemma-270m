#!/usr/bin/env python3
"""Download checkpoint from HuggingFace Hub"""
import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("huggingface_hub not installed")
    exit(0)

username = os.environ.get('HF_USERNAME', '')
model_repo = f'{username}/multimodal-gemma-270m-checkpoints'

try:
    checkpoint_path = hf_hub_download(
        repo_id=model_repo,
        filename='final_model.ckpt',
        local_dir='models/checkpoints/gemma-270m-llava-training'
    )
    print(f'Downloaded checkpoint from {model_repo}')
except Exception as e:
    print(f'No checkpoint on HF Hub: {e}')
    print('Will use dummy model for deployment demo')
