#!/usr/bin/env python3
"""
Upload Multimodal Gemma Model to Hugging Face
"""
import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import sys

def upload_model():
    # Configuration
    MODEL_NAME = "multimodal-gemma-270m-llava"  # Change this to your desired name
    USERNAME = "YOUR_USERNAME"  # Replace with your HF username

    print("🚀 Uploading Multimodal Gemma Model to Hugging Face")
    print("=" * 60)

    # Check if logged in
    api = HfApi()
    try:
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
        USERNAME = user_info['name']
    except Exception as e:
        print("❌ Not logged in to Hugging Face!")
        print("Please run: huggingface-cli login")
        return False

    repo_id = f"{USERNAME}/{MODEL_NAME}"

    # Create repository
    print(f"\n📝 Creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False  # Set to True if you want private repo
        )
        print(f"✅ Repository created: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Repository might already exist: {e}")

    # Upload files
    print(f"\n📤 Uploading model files...")

    # Upload model checkpoint
    checkpoint_path = "models/checkpoints/gemma-270m-llava-training/final_model.ckpt"
    if Path(checkpoint_path).exists():
        print("📁 Uploading model checkpoint (1.2GB)...")
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo="final_model.ckpt",
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ Model checkpoint uploaded!")
    else:
        print("❌ Model checkpoint not found!")
        return False

    # Upload config files
    config_files = [
        "configs/model_config.yaml",
        "configs/training_config.yaml",
        "configs/data_config.yaml"
    ]

    for config_file in config_files:
        if Path(config_file).exists():
            print(f"📁 Uploading {config_file}...")
            api.upload_file(
                path_or_fileobj=config_file,
                path_in_repo=f"configs/{Path(config_file).name}",
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"✅ {config_file} uploaded!")

    # Create README
    readme_content = f"""# Multimodal Gemma 270M - LLaVA Architecture

This is a multimodal vision-language model based on Google's Gemma-270M, trained using the LLaVA architecture.

## Model Details

- **Base Model**: Google Gemma-270M (270 million parameters)
- **Vision Encoder**: CLIP ViT-Large/14@336px
- **Architecture**: LLaVA-style vision-language fusion
- **Training**: 7 epochs on LLaVA-150K dataset
- **Trainable Parameters**: 18.6M / 539M total
- **Quantization**: 4-bit with LoRA fine-tuning

## Usage

```python
from src.models import MultimodalGemmaLightning
from src.utils.config import load_config, merge_configs

# Load config
model_config = load_config("configs/model_config.yaml")
training_config = load_config("configs/training_config.yaml")
data_config = load_config("configs/data_config.yaml")
config = merge_configs([model_config, training_config, data_config])

# Load model
model = MultimodalGemmaLightning.load_from_checkpoint(
    "final_model.ckpt",
    config=config,
    strict=False
)
model.eval()
```

## Training Details

- **Dataset**: LLaVA-150K instruction tuning dataset
- **Training Time**: ~12 hours on A100 GPU
- **Loss**: Converged from 3.3 to stable training loss
- **Vision-Language Fusion**: Image tokens replace <image> placeholders

## Capabilities

This model can:
- Process images and answer questions about them
- Describe visual content in images
- Follow vision-language instructions
- Generate relevant text based on image content

Note: As a small model with limited training, responses may be simple but are contextually relevant to the input images.

## Demo

Try the live demo: [Gradio Space](https://huggingface.co/spaces/{USERNAME}/{MODEL_NAME}-demo)
"""

    print("📁 Creating README...")
    with open("/tmp/README.md", "w") as f:
        f.write(readme_content)

    api.upload_file(
        path_or_fileobj="/tmp/README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    print("✅ README uploaded!")

    print(f"\n🎉 SUCCESS! Model uploaded to:")
    print(f"🔗 https://huggingface.co/{repo_id}")

    print(f"\n📋 Next Steps:")
    print(f"1. Create Gradio Space: https://huggingface.co/new-space")
    print(f"2. Use repo_id: {repo_id}")
    print(f"3. Upload your gradio_app.py to the Space")

    return repo_id

if __name__ == "__main__":
    repo_id = upload_model()
    if repo_id:
        print(f"\n✅ Ready to create Gradio Space with model: {repo_id}")