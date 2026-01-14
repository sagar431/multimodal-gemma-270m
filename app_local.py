#!/usr/bin/env python3
"""
Local Gradio App for Multimodal Gemma-270M Inference
Uses the same model loading as inference.py (which works!)
"""

import os
import sys
import logging
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.multimodal_gemma import MultimodalGemma

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Model paths
CHECKPOINT_PATH = "models/checkpoints/gemma-270m-llava-a100-optimized/final_model.ckpt"

# Global cache
_model = None
_tokenizer = None
_image_processor = None
_device = None


def get_default_config():
    """Get default configuration"""
    return OmegaConf.create({
        "model": {
            "gemma_model_name": "google/gemma-3-270m",
            "vision_model_name": "openai/clip-vit-large-patch14",
            "projector_hidden_dim": 2048,
            "use_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "use_nested_quant": False,
            "lora": {
                "r": 64,
                "alpha": 128,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            }
        },
        "special_tokens": {
            "image_token": "<image>",
            "pad_token": "<pad>"
        },
        "tokenizer": {
            "padding_side": "right",
            "truncation": True,
            "max_length": 512,
            "add_special_tokens": True
        }
    })


def load_model():
    """Load the trained multimodal model (same as inference.py)"""
    global _model, _tokenizer, _image_processor, _device
    
    if _model is not None:
        return _model, _tokenizer, _image_processor
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {_device}")
    
    # Load checkpoint
    checkpoint_path = project_root / CHECKPOINT_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    log.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    
    # Get config from checkpoint
    if 'hyper_parameters' in checkpoint:
        config = checkpoint['hyper_parameters'].get('config', None)
    else:
        config = None
    
    if config is None:
        log.warning("No config found in checkpoint, using defaults")
        config = get_default_config()
    
    # Initialize model using the same class as inference.py
    log.info("Initializing MultimodalGemma model...")
    model = MultimodalGemma(config)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('model.'):
                state_dict[k[6:]] = v
            else:
                state_dict[k] = v
        model.load_state_dict(state_dict, strict=False)
        log.info("✅ Loaded model weights")
    
    model.to(_device)
    model.eval()
    
    # Get processors from model
    _model = model
    _tokenizer = model.tokenizer
    _image_processor = model.vision_processor
    
    log.info("✅ Model loaded successfully!")
    return _model, _tokenizer, _image_processor


def predict(image, question, max_tokens, temperature):
    """Generate response for image + question"""
    
    if image is None:
        return "⚠️ Please upload an image first!"
    
    if not question or not question.strip():
        question = "What do you see in this image?"
    
    try:
        model, tokenizer, image_processor = load_model()
        
        # Process image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        else:
            image = image.convert('RGB')
        
        # Process image - same as inference.py
        image_inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(_device)
        
        # Format prompt with image token
        formatted_prompt = f"<image>\nHuman: {question}\nAssistant:"
        
        # Tokenize
        text_inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = text_inputs["input_ids"].to(_device)
        attention_mask = text_inputs["attention_mask"].to(_device)
        
        # Generate using model's built-in method (same as inference.py)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=0.9,
                do_sample=True,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response if response else "I see the image but couldn't generate a response."
    
    except Exception as e:
        log.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


# Build Gradio interface
with gr.Blocks(title="🤖 Multimodal Gemma-270M") as demo:
    
    gr.Markdown("""
    # 🤖 Multimodal Gemma-270M - Local Inference
    
    Upload an image and ask questions about it!
    
    **Model**: Gemma-3-270M + CLIP Vision Encoder + LoRA fine-tuning on LLaVA data
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📷 Upload Image",
                type="pil",
                height=400
            )
            question_input = gr.Textbox(
                label="❓ Your Question",
                placeholder="What do you see in this image?",
                value="What do you see in this image?",
                lines=2
            )
            
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=300,
                    value=100,
                    step=10,
                    label="🔢 Max Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="🌡️ Temperature"
                )
            
            submit_btn = gr.Button("🚀 Generate Response", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output = gr.Textbox(
                label="🤖 Model Response",
                lines=15,
                max_lines=20
            )
    
    # Example prompts
    gr.Markdown("### 💡 Example Questions to Try")
    gr.Markdown("""
    - "What do you see in this image?"
    - "Describe the main objects in detail."
    - "What colors are visible?"
    - "Is there a person in this image? What are they doing?"
    - "Describe the setting or environment."
    """)
    
    # Wire up the button
    submit_btn.click(
        fn=predict,
        inputs=[image_input, question_input, max_tokens, temperature],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    **Tips:**
    - Lower temperature (0.1-0.5) = more focused/deterministic responses
    - Higher temperature (0.8-1.5) = more creative/varied responses
    - Increase max tokens for longer descriptions
    """)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Multimodal Gemma-270M Local Inference App")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
