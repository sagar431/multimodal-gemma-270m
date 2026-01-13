---
title: Multimodal Gemma-270M
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# Multimodal Gemma-270M

A vision-language model built with Google's Gemma-270M and CLIP vision encoder, trained on LLaVA-150K dataset.

## Features

- **Vision-Language Understanding**: Upload an image and ask questions about it
- **LLaVA Architecture**: Combines language model with vision encoder via learned projector
- **Interactive Chat**: Multi-turn conversation about images

## Model Details

| Component | Details |
|-----------|---------|
| Language Model | Google Gemma-270M |
| Vision Encoder | CLIP ViT-Large/14 |
| Training Data | LLaVA-150K + COCO |
| Framework | PyTorch Lightning |

## Usage

1. Click "Load Model" to initialize
2. Upload an image
3. Ask questions about the image
4. Adjust generation settings as needed

## MLOps Pipeline

This model is deployed via an automated CI/CD pipeline:
- Automated testing on push
- Model training with PyTorch Lightning
- Automatic deployment to HuggingFace Spaces

## License

Apache 2.0
