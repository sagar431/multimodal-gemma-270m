#!/usr/bin/env python3
"""
Gradio UI for Multimodal Gemma Model
"""
import sys
import torch
import gradio as gr
from pathlib import Path
from PIL import Image
import io
import time
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.models import MultimodalGemmaLightning
from src.utils.config import load_config, merge_configs

# Global model variable
model = None
config = None

def load_model():
    """Load the trained multimodal model"""
    global model, config

    if model is not None:
        return "✅ Model already loaded!"

    try:
        print("🔄 Loading multimodal Gemma model...")

        # Load config
        model_config = load_config("configs/model_config.yaml")
        training_config = load_config("configs/training_config.yaml")
        data_config = load_config("configs/data_config.yaml")
        config = merge_configs([model_config, training_config, data_config])

        # Load model from checkpoint
        checkpoint_path = "models/checkpoints/gemma-270m-llava-training/final_model.ckpt"

        if not Path(checkpoint_path).exists():
            return "❌ Model checkpoint not found! Please train the model first."

        model = MultimodalGemmaLightning.load_from_checkpoint(
            checkpoint_path,
            config=config,
            strict=False,
            map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
        model.eval()

        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        print(f"✅ Model loaded successfully on {device}!")
        return f"✅ Model loaded successfully on {device}!"

    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        print(error_msg)
        return error_msg

def predict_with_image(image, question, max_tokens=100, temperature=0.7):
    """Generate response for image + text input"""
    global model, config

    if model is None:
        return "❌ Please load the model first using the 'Load Model' button!"

    if image is None:
        return "❌ Please upload an image!"

    if not question.strip():
        question = "What do you see in this image?"

    try:
        # Get device
        device = next(model.parameters()).device

        # Process image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')

        # Prepare image for model
        vision_inputs = model.model.vision_processor(
            images=[image],
            return_tensors="pt"
        )
        pixel_values = vision_inputs["pixel_values"].to(device)

        # Prepare text prompt
        prompt = f"<image>\nHuman: {question}\nAssistant:"

        # Tokenize text
        text_inputs = model.model.tokenizer(
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
            # Use the full multimodal model with image inputs
            outputs = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                max_new_tokens=min(max_tokens, 150),
                temperature=min(max(temperature, 0.1), 2.0),
                do_sample=temperature > 0.1,
                repetition_penalty=1.1
            )

        # Decode response
        input_length = input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response = model.model.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up response
        response = response.strip()
        if not response:
            response = "I can see the image, but I'm having trouble generating a detailed response."

        return response

    except Exception as e:
        error_msg = f"❌ Error during inference: {str(e)}"
        print(error_msg)
        return error_msg

def chat_with_image(image, question, history, max_tokens, temperature):
    """Chat interface function"""
    if model is None:
        response = "❌ Please load the model first!"
    else:
        response = predict_with_image(image, question, max_tokens, temperature)

    # Add to history - using messages format
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})
    return history, ""

def load_example_image():
    """Load an example image from cache if available"""
    cache_dir = Path("data/cache/images/")
    if cache_dir.exists():
        image_files = list(cache_dir.glob("*.jpg"))
        if image_files:
            return str(image_files[0])
    return None

def create_gradio_interface():
    """Create the Gradio interface"""

    # Custom CSS for better styling
    css = """
    .container {
        max-width: 1200px;
        margin: auto;
        padding: 20px;
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
    .model-info {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """

    with gr.Blocks(css=css, title="Multimodal Gemma Chat") as demo:
        gr.HTML("""
        <div class="header">
            <h1>🎉 Multimodal Gemma-270M Chat</h1>
            <p>Upload an image and chat with your trained vision-language model!</p>
        </div>
        """)

        # Model status section
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="model-info">
                    <h3>📊 Model Info</h3>
                    <ul>
                        <li><strong>Base Model:</strong> Google Gemma-270M</li>
                        <li><strong>Vision:</strong> CLIP ViT-Large</li>
                        <li><strong>Training:</strong> LLaVA-150K + COCO Images</li>
                        <li><strong>Parameters:</strong> 18.6M trainable / 539M total</li>
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

        gr.HTML("<hr>")

        # Main interface
        with gr.Row():
            # Left column - Image and controls
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📸 Upload Image",
                    type="pil",
                    height=300
                )

                # Example images
                gr.HTML("<p><strong>💡 Tip:</strong> Upload any image or use the 'Load Example' button</p>")
                example_btn = gr.Button("🎲 Load Example Image")

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

            for i, question in enumerate(example_questions):
                if i % 3 == 0:
                    with gr.Row():
                        pass
                gr.Button(
                    question,
                    size="sm"
                ).click(
                    lambda x=question: x,
                    outputs=question_input
                )

        # Footer
        gr.HTML("""
        <hr>
        <div style="text-align: center; margin-top: 20px;">
            <p><strong>🎯 Your Multimodal Gemma Model</strong></p>
            <p>Text-only → Vision-Language Model using LLaVA Architecture</p>
        </div>
        """)

        # Event handlers
        load_btn.click(
            fn=load_model,
            outputs=model_status
        )

        example_btn.click(
            fn=load_example_image,
            outputs=image_input
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

    return demo

def main():
    """Main function to launch the Gradio app"""
    print("🚀 Starting Multimodal Gemma Gradio App...")

    # Create interface
    demo = create_gradio_interface()

    # Launch
    print("🌐 Launching Gradio interface...")
    print("📱 Access at: http://localhost:7860")

    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public link
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()