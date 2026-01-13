# Multimodal Gemma-270M MLOps Project

[![Train & Deploy](https://github.com/YOUR_USERNAME/multimodal-gemma-270m/actions/workflows/train_deploy.yml/badge.svg)](https://github.com/YOUR_USERNAME/multimodal-gemma-270m/actions/workflows/train_deploy.yml)
[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/multimodal-gemma-270m)

A production-ready **Multimodal Vision-Language Model** built with PyTorch Lightning and automated MLOps CI/CD pipeline for deployment to HuggingFace Spaces.

## 🌟 Features

- **Multimodal Architecture**: Combines Google Gemma-270M with CLIP vision encoder
- **PyTorch Lightning**: Clean, modular training code with automatic optimization
- **MLOps Pipeline**: Automated CI/CD with GitHub Actions
- **DVC Integration**: Data versioning and pipeline orchestration
- **Auto-Deployment**: Push to main → Test → Train → Deploy to HuggingFace Spaces
- **Gradio Interface**: Beautiful, interactive web UI for inference

## 📁 Project Structure

```
multimodal-gemma-270m/
├── .github/
│   └── workflows/
│       └── train_deploy.yml    # CI/CD pipeline
├── configs/
│   ├── config.yaml            # Main Hydra config
│   ├── model_config.yaml      # Model architecture
│   ├── training_config.yaml   # Training hyperparameters
│   └── data_config.yaml       # Dataset configuration
├── src/
│   ├── models/
│   │   ├── lightning_module.py   # PyTorch Lightning module
│   │   ├── multimodal_gemma.py   # Core model architecture
│   │   └── projectors.py         # Vision/Audio projectors
│   ├── data/
│   │   └── datamodule.py         # Lightning DataModule
│   ├── utils/
│   │   └── config.py             # Configuration utilities
│   └── trace_model.py            # Model export for deployment
├── hf_space/
│   ├── app.py                 # Gradio app for HuggingFace Spaces
│   ├── requirements.txt       # Space dependencies
│   └── README.md              # Space metadata
├── scripts/
│   ├── prepare_data.py        # Data preparation
│   └── validate_model.py      # Model validation
├── tests/
│   ├── test_model.py          # Model unit tests
│   └── test_app.py            # App tests
├── train.py                   # Main training script
├── gradio_app.py              # Local Gradio app
├── dvc.yaml                   # DVC pipeline definition
├── pyproject.toml             # Project configuration
├── Dockerfile                 # Container definition
├── Makefile                   # Convenience commands
└── README.md                  # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/multimodal-gemma-270m.git
cd multimodal-gemma-270m

# Install dependencies
pip install -r requirements.txt

# Or use make
make install
```

### Training

```bash
# Full training
python train.py

# Quick test run
python train.py trainer.fast_dev_run=true

# With Weights & Biases logging
python train.py logging.use_wandb=true

# Or use make
make train
make train-fast
```

### Model Export & Deployment

```bash
# Export model for deployment
python src/trace_model.py --output_path hf_space/model.pt

# Validate exported model
python scripts/validate_model.py --model_path hf_space/model.pt

# Or use make
make trace
make validate
```

### Local Inference

```bash
# Run Gradio locally
cd hf_space && python app.py

# Or use the full local app
python gradio_app.py
```

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for automated MLOps:

```
Push to main → Tests → Train (optional) → Trace → Deploy to HuggingFace Spaces
```

### Pipeline Jobs

1. **Test**: Runs linting and unit tests
2. **Train**: Trains the model (manual trigger or on demand)
3. **Trace**: Exports model to deployment format
4. **Deploy**: Uploads to HuggingFace Spaces
5. **Integration Test**: Verifies deployed space

### GitHub Secrets Required

Set these in your repository settings:

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | HuggingFace API token with write access |
| `HF_USERNAME` | Your HuggingFace username |
| `WANDB_API_KEY` | (Optional) Weights & Biases API key |

### Manual Deployment

Trigger a deployment manually:

```bash
# Via GitHub CLI
gh workflow run train_deploy.yml

# With training
gh workflow run train_deploy.yml -f run_training=true -f max_epochs=5
```

## 📊 DVC Pipeline

Use DVC for reproducible ML pipelines:

```bash
# Run full pipeline
dvc repro

# Run specific stage
dvc repro train

# Visualize pipeline
dvc dag
```

### Pipeline Stages

```
prepare_data → train → trace → validate
```

## 🏗️ Architecture

### Model Components

- **Language Model**: Google Gemma-270M with LoRA adapters
- **Vision Encoder**: CLIP ViT-Large/14
- **Vision Projector**: MLP connecting vision to language
- **Training**: LLaVA-style multimodal instruction tuning

### Key Parameters

| Component | Size |
|-----------|------|
| Language Model | 270M parameters |
| Vision Encoder | 428M parameters |
| Trainable (LoRA + Projector) | ~18.6M parameters |

## 📝 Configuration

Modify configs via Hydra:

```bash
# Change model
python train.py model.gemma_model_name=google/gemma-2b

# Change training
python train.py training.max_epochs=10 training.projector_lr=1e-4

# Use different experiment
python train.py experiment=my_experiment
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Or use make
make test
make test-cov
```

## 🐳 Docker

```bash
# Build image
docker build -t multimodal-gemma:latest .

# Train with GPU
docker run --gpus all -v $(pwd)/models:/app/models multimodal-gemma:latest

# Interactive shell
docker run --gpus all -it multimodal-gemma:latest bash
```

## 📚 References

- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [Gemma Technical Report](https://arxiv.org/abs/2403.08295)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [HuggingFace Spaces](https://huggingface.co/docs/hub/spaces)

## 📄 License

Apache 2.0

## 🙏 Acknowledgments

- Google for Gemma models
- OpenAI for CLIP
- LLaVA team for multimodal architecture inspiration
- PyTorch Lightning team for the training framework