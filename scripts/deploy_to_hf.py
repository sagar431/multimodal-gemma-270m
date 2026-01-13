#!/usr/bin/env python3
"""Deploy to HuggingFace Spaces"""
import os
import sys

try:
    from huggingface_hub import HfApi, upload_folder
except ImportError:
    print("huggingface_hub not installed")
    sys.exit(1)

api = HfApi()
username = os.environ.get('HF_USERNAME', 'your-username')
space_name = os.environ.get('HF_SPACE_NAME', 'multimodal-gemma-270m')
space_id = f'{username}/{space_name}'

print(f'Deploying to: {space_id}')

# Create space if it doesn't exist
try:
    api.create_repo(
        repo_id=space_id,
        repo_type='space',
        space_sdk='gradio',
        exist_ok=True
    )
    print(f'Space {space_id} is ready!')
except Exception as e:
    print(f'Space creation note: {e}')

# Upload all files from hf_space folder
upload_folder(
    folder_path='hf_space',
    repo_id=space_id,
    repo_type='space',
    commit_message='Deploy Multimodal Gemma-270M via CI/CD'
)
print('')
print('=========================================')
print(f'Deployed to https://huggingface.co/spaces/{space_id}')
print('=========================================')
