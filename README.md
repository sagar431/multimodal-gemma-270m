# Multimodal Gemma-270M: Vision-Language Model

A multimodal vision-language model based on Google's Gemma-270M, trained using the LLaVA architecture to understand and generate text responses about images.

## 🎯 Project Overview

This project transforms Google's text-only Gemma-270M model into a multimodal vision-language model capable of:
- Processing images and answering questions about them
- Describing visual content in natural language
- Following vision-language instructions
- Understanding relationships between images and text

## 🏗️ Architecture

### Core Components

- **Base Language Model**: Google Gemma-270M (270 million parameters)
- **Vision Encoder**: CLIP ViT-Large/14@336px
- **Vision-Language Fusion**: LLaVA-style architecture with trainable projector
- **Training Method**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Quantization**: 4-bit quantization with BitsAndBytes

### Model Statistics
- **Total Parameters**: 539M
- **Trainable Parameters**: 18.6M (3.4%)
- **Training Data**: LLaVA-150K instruction tuning dataset
- **Training Time**: ~12 hours on A100 GPU
- **Final Model Size**: 1.2GB

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.9+
CUDA-capable GPU (recommended)
16GB+ RAM
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/sagar431/multimodal-gemma-270m.git
cd multimodal-gemma-270m
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download pre-trained model** (if available):
```bash
# The model will be downloaded automatically when running inference
python quick_test.py
```

## 🔧 Usage

### Local Inference

1. **Quick Test**:
```bash
python quick_test.py
```

2. **Simple Verification**:
```bash
python simple_test.py
```

3. **Gradio Web Interface**:
```bash
python gradio_app.py
# Open http://localhost:7860 in your browser
```

### Training from Scratch

1. **Prepare the training data**:
```bash
python scripts/prepare_data.py
```

2. **Start training**:
```bash
python train.py
```

3. **Monitor training** (if using wandb):
```bash
# Check your wandb dashboard for training metrics
```

### Hugging Face Integration

The trained model is available on Hugging Face:

- **Model Repository**: [sagar007/multimodal-gemma-270m-llava](https://huggingface.co/sagar007/multimodal-gemma-270m-llava)
- **Live Demo**: [Gradio Space](https://huggingface.co/spaces/sagar007/multimodal-gemma-270m-demo)

```python
from huggingface_hub import hf_hub_download
from src.models import MultimodalGemmaLightning
from src.utils.config import load_config, merge_configs

# Download and load model
checkpoint_path = hf_hub_download(
    repo_id="sagar007/multimodal-gemma-270m-llava",
    filename="final_model.ckpt"
)

# Load configs
model_config = load_config("configs/model_config.yaml")
training_config = load_config("configs/training_config.yaml")
data_config = load_config("configs/data_config.yaml")
config = merge_configs([model_config, training_config, data_config])

# Load model
model = MultimodalGemmaLightning.load_from_checkpoint(
    checkpoint_path,
    config=config,
    strict=False
)
model.eval()
```

## 📁 Project Structure

```
multimodal-gemma-270m/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── multimodal_gemma.py      # Main multimodal model
│   │   └── projectors.py            # Vision-language projectors
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py              # Dataset classes
│   │   └── image_utils.py           # Image processing utilities
│   ├── training/
│   │   ├── __init__.py
│   │   └── lightning_trainer.py     # PyTorch Lightning trainer
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       └── logging.py               # Logging utilities
├── configs/
│   ├── model_config.yaml            # Model architecture config
│   ├── training_config.yaml         # Training hyperparameters
│   └── data_config.yaml             # Dataset configuration
├── scripts/
│   ├── prepare_data.py              # Data preparation
│   ├── download_coco.py             # COCO image downloader
│   └── upload_to_hf.py              # Hugging Face upload script
├── gradio_app.py                    # Local Gradio interface
├── space_app.py                     # Hugging Face Space version
├── train.py                         # Main training script
├── quick_test.py                    # Quick model test
├── simple_test.py                   # Simple checkpoint verification
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🔬 Technical Details

### Vision-Language Fusion Process

1. **Image Processing**: Images are processed through CLIP ViT-Large encoder
2. **Feature Projection**: Vision features are projected to language model embedding space
3. **Token Replacement**: Image features replace `<image>` tokens in the input sequence
4. **Multimodal Generation**: The fused model generates text responses

### Training Process

1. **Data Loading**: LLaVA-150K dataset with instruction-following conversations
2. **Image Download**: COCO images are cached locally for training
3. **LoRA Fine-tuning**: Only vision projector and LoRA adapters are trained
4. **Mixed Precision**: bf16 training for efficiency
5. **Gradient Accumulation**: Batch size optimization for memory efficiency

### Key Features

- **Memory Efficient**: 4-bit quantization reduces memory usage
- **Fast Training**: LoRA adapters enable quick fine-tuning
- **Scalable**: Modular architecture for easy extension
- **Production Ready**: Includes deployment scripts and interfaces

## 📊 Results

### Training Metrics
- **Initial Loss**: 3.3
- **Final Loss**: Stable convergence
- **Training Epochs**: 7
- **GPU Hours**: ~12 on A100

### Performance
- **Response Quality**: Contextually relevant to input images
- **Vision Understanding**: Basic object and scene recognition
- **Instruction Following**: Follows simple vision-language prompts

### Limitations
- **Model Size**: Small model may produce simple responses
- **Training Data**: Limited to LLaVA-150K dataset scope
- **Quantization**: 4-bit precision may affect generation quality

## 🛠️ Development

### Training Your Own Model

1. **Modify configs** in `configs/` directory
2. **Customize datasets** in `src/data/datasets.py`
3. **Adjust model architecture** in `src/models/multimodal_gemma.py`
4. **Run training**: `python train.py`

### Adding New Features

- **Audio Modality**: Extend with audio encoders (framework ready)
- **Different Base Models**: Swap Gemma with other language models
- **Custom Datasets**: Add your own vision-language datasets
- **Advanced Projectors**: Implement more sophisticated fusion methods

## 🔍 Debugging

### Common Issues

1. **CUDA Out of Memory**:
```bash
# Reduce batch size in training_config.yaml
batch_size: 1
gradient_accumulation_steps: 4
```

2. **Model Loading Errors**:
```bash
# Verify checkpoint path and config files
python simple_test.py
```

3. **PEFT Conflicts**:
```bash
# Ensure compatible versions
pip install peft>=0.6.0
```

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{multimodal-gemma-270m,
  title={Multimodal Gemma-270M: Vision-Language Model with LLaVA Architecture},
  author={Sagar},
  year={2024},
  url={https://github.com/sagar431/multimodal-gemma-270m}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google**: For the Gemma-270M base model
- **OpenAI**: For the CLIP vision encoder
- **LLaVA Team**: For the vision-language architecture
- **Hugging Face**: For the transformers and model hosting
- **PyTorch Lightning**: For the training framework

## 🔗 Links

- **Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/sagar007/multimodal-gemma-270m-demo)
- **Model Weights**: [Hugging Face Model](https://huggingface.co/sagar007/multimodal-gemma-270m-llava)
- **GitHub Repository**: [sagar431/multimodal-gemma-270m](https://github.com/sagar431/multimodal-gemma-270m)

---

## 📞 Support

For questions, issues, or collaboration:
- Open an issue on GitHub
- Check the Hugging Face model page
- Review the live demo for examples

**Happy multimodal modeling! 🎉**