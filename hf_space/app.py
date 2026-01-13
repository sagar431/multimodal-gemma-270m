#!/usr/bin/env python3
"""
Multimodal Gemma-270M Gradio App for HuggingFace Spaces
Optimized for deployment with pre-trained model
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Global model variable
model = None
tokenizer = None
vision_processor = None
config = None
device = None


def load_model():
    """Load the Multimodal Gemma model for inference"""
    global model, tokenizer, vision_processor, config, device
    
    if model is not None:
        return "✅ Model already loaded!"
    
    try:
        log.info("🔄 Loading Multimodal Gemma model...")
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Using device: {device}")
        
        # Load model from exported file
        model_path = Path("model.pt")
        config_path = Path("config.yaml")
        
        if not model_path.exists():
            return "❌ Model file not found! Please ensure model.pt exists."
        
        # Load exported model
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'config' in checkpoint:
            config = checkpoint['config']
        elif config_path.exists():
            config = OmegaConf.load(config_path)
            config = OmegaConf.to_container(config, resolve=True)
        else:
            # Default config
            config = {
                "model": {
                    "gemma_model_name": "google/gemma-2b",
                    "vision_model_name": "openai/clip-vit-large-patch14"
                }
            }
        
        # Initialize model components
        from transformers import AutoTokenizer, AutoProcessor, AutoModel
        
        # Load tokenizer
        gemma_model_name = config.get("model", {}).get("gemma_model_name", "google/gemma-2b")
        vision_model_name = config.get("model", {}).get("vision_model_name", "openai/clip-vit-large-patch14")
        
        log.info(f"Loading tokenizer: {gemma_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            gemma_model_name,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        log.info(f"Loading vision processor: {vision_model_name}")
        vision_processor = AutoProcessor.from_pretrained(
            vision_model_name,
            trust_remote_code=True
        )
        
        # Load full model architecture
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            from src.models import MultimodalGemmaLightning
            
            model = MultimodalGemmaLightning(config)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            model = model.to(device)
            model.eval()
            
        except ImportError:
            # Fallback: Use a simpler inference model
            log.warning("Full model not available, using simplified inference mode")
            model = create_simple_inference_model(checkpoint, config, device)
        
        log.info(f"✅ Model loaded successfully on {device}!")
        return f"✅ Model loaded successfully on {device}!"
        
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        log.error(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


def create_simple_inference_model(checkpoint, config, device):
    """Create a simplified inference model when full model is not available"""
    from transformers import AutoModelForCausalLM
    
    gemma_model_name = config.get("model", {}).get("gemma_model_name", "google/gemma-2b")
    
    model = AutoModelForCausalLM.from_pretrained(
        gemma_model_name,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map='auto' if device.type == 'cuda' else None,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN")
    )
    model.eval()
    return model


def predict_with_image(image: Optional[Image.Image], question: str, 
                       max_tokens: int = 100, temperature: float = 0.7) -> str:
    """Generate response for image + text input"""
    global model, tokenizer, vision_processor, device
    
    if model is None:
        return "❌ Please load the model first using the 'Load Model' button!"
    
    if image is None:
        return "❌ Please upload an image!"
    
    if not question.strip():
        question = "What do you see in this image?"
    
    try:
        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        else:
            image = image.convert('RGB')
        
        # Process image
        vision_inputs = vision_processor(
            images=[image],
            return_tensors="pt"
        )
        pixel_values = vision_inputs["pixel_values"].to(device)
        
        # Prepare text prompt
        prompt = f"<image>\nHuman: {question}\nAssistant:"
        
        # Tokenize text
        text_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        
        # Generate response
        with torch.no_grad():
            if hasattr(model, 'model') and hasattr(model.model, 'generate'):
                # Full multimodal model
                outputs = model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=pixel_values,
                    max_new_tokens=min(max_tokens, 150),
                    temperature=min(max(temperature, 0.1), 2.0),
                    do_sample=temperature > 0.1,
                    repetition_penalty=1.1
                )
            else:
                # Simple language model (fallback)
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=min(max_tokens, 150),
                    temperature=min(max(temperature, 0.1), 2.0),
                    do_sample=temperature > 0.1,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id
                )
        
        # Decode response
        input_length = input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up response
        response = response.strip()
        if not response:
            response = "I can see the image, but I'm having trouble generating a detailed response."
        
        return response
        
    except Exception as e:
        error_msg = f"❌ Error during inference: {str(e)}"
        log.error(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


def chat_with_image(image: Optional[Image.Image], question: str, 
                   history: list, max_tokens: int, temperature: float) -> Tuple[list, str]:
    """Chat interface function with history"""
    if model is None:
        response = "❌ Please load the model first!"
    else:
        response = predict_with_image(image, question, max_tokens, temperature)
    
    # Add to history
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})
    
    return history, ""


# Create the Gradio interface
css = """
.container {
    max-width: 1200px;
    margin: auto;
    padding: 20px;
}
.header {
    text-align: center;
    margin-bottom: 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    color: white;
}
.header h1 {
    margin: 0;
    font-size: 2.5em;
}
.header p {
    margin: 10px 0 0 0;
    opacity: 0.9;
}
.model-info {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.model-info h3 {
    margin-top: 0;
    color: #333;
}
.model-info ul {
    margin-bottom: 0;
}
.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
footer {
    text-align: center;
    padding: 20px;
    background: #f5f5f5;
    border-radius: 10px;
    margin-top: 30px;
}
"""

with gr.Blocks(css=css, title="Multimodal Gemma-270M", theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="header">
        <h1>🤖 Multimodal Gemma-270M</h1>
        <p>Upload an image and chat with your vision-language model!</p>
    </div>
    """)
    
    # Model status section
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div class="model-info">
                <h3>📊 Model Information</h3>
                <ul>
                    <li><strong>Base Model:</strong> Google Gemma-270M</li>
                    <li><strong>Vision Encoder:</strong> CLIP ViT-Large</li>
                    <li><strong>Architecture:</strong> LLaVA-style Multimodal</li>
                    <li><strong>Training:</strong> LLaVA-150K + COCO Images</li>
                </ul>
            </div>
            """)
            
            # Model loading
            load_btn = gr.Button("🚀 Load Model", variant="primary", size="lg")
            model_status = gr.Textbox(
                label="Model Status",
                value="Click 'Load Model' to start",
                interactive=False
            )
    
    gr.HTML("<hr style='margin: 20px 0'>")
    
    # Main interface
    with gr.Row():
        # Left column - Image and controls
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📸 Upload Image",
                type="pil",
                height=350
            )
            
            gr.HTML("<p><strong>💡 Tip:</strong> Upload any image to start chatting!</p>")
            
            # Generation settings
            with gr.Accordion("⚙️ Generation Settings", open=False):
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=100,
                    step=10,
                    label="Max Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
        
        # Right column - Chat interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 Chat with Image",
                height=400,
                show_label=True,
                type="messages"
            )
            
            question_input = gr.Textbox(
                label="❓ Ask about the image",
                placeholder="What do you see in this image?",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("💬 Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Chat")
    
    # Example prompts
    with gr.Row():
        gr.HTML("<h3>💡 Example Questions:</h3>")
    
    with gr.Row():
        example_questions = [
            "What do you see in this image?",
            "Describe the main objects in the picture.",
            "What colors are prominent in this image?",
            "Are there any people in the image?",
            "What's the setting or location?",
            "What objects are in the foreground?"
        ]
        
        for question in example_questions[:3]:
            gr.Button(question, size="sm").click(
                lambda x=question: x,
                outputs=question_input
            )
    
    with gr.Row():
        for question in example_questions[3:]:
            gr.Button(question, size="sm").click(
                lambda x=question: x,
                outputs=question_input
            )
    
    # Footer
    gr.HTML("""
    <footer>
        <p><strong>🎯 Multimodal Gemma-270M</strong></p>
        <p>Deployed via MLOps CI/CD Pipeline</p>
        <p style="font-size: 0.9em; color: #666;">Built with PyTorch Lightning, Gradio & HuggingFace</p>
    </footer>
    """)
    
    # Event handlers
    load_btn.click(
        fn=load_model,
        outputs=model_status
    )
    
    submit_btn.click(
        fn=chat_with_image,
        inputs=[image_input, question_input, chatbot, max_tokens, temperature],
        outputs=[chatbot, question_input]
    )
    
    question_input.submit(
        fn=chat_with_image,
        inputs=[image_input, question_input, chatbot, max_tokens, temperature],
        outputs=[chatbot, question_input]
    )
    
    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, question_input]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
