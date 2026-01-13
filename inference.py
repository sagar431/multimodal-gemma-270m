#!/usr/bin/env python3
"""
Inference script for Multimodal Gemma - generates predictions on test images
"""
import os
import sys
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.multimodal_gemma import MultimodalGemma
from omegaconf import OmegaConf
from transformers import AutoTokenizer, CLIPImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalGemmaInference:
    """Inference wrapper for Multimodal Gemma model"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        """Initialize inference model
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config from checkpoint or load from file
        if 'hyper_parameters' in checkpoint:
            self.config = checkpoint['hyper_parameters'].get('config', None)
        else:
            self.config = None
            
        if self.config is None:
            if config_path and Path(config_path).exists():
                self.config = OmegaConf.load(config_path)
            else:
                # Use default config
                logger.warning("No config found, using defaults")
                self.config = self._get_default_config()
        
        # Initialize model
        logger.info("Initializing model...")
        self.model = MultimodalGemma(self.config)
        
        # Load state dict
        if 'state_dict' in checkpoint:
            # Remove 'model.' prefix if present
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model.'):
                    state_dict[k[6:]] = v
                else:
                    state_dict[k] = v
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get processors
        self.tokenizer = self.model.tokenizer
        self.image_processor = self.model.vision_processor
        
        logger.info("Model loaded successfully!")
    
    def _get_default_config(self):
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
    
    @torch.no_grad()
    def generate(
        self,
        image_path: str,
        prompt: str = "What do you see in this image?",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text description for an image

        Args:
            image_path: Path to image file
            prompt: Text prompt/question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        # Load and process image
        image = Image.open(image_path).convert('RGB')

        # Process image
        image_inputs = self.image_processor(
            images=image,
            return_tensors="pt"
        )
        pixel_values = image_inputs["pixel_values"].to(self.device)

        # Format prompt with image token
        image_token = self.config.get("special_tokens", {}).get("image_token", "<image>")
        formatted_prompt = f"{image_token}\nHuman: {prompt}\nAssistant:"

        # Tokenize
        text_inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device)

        # Use the model's built-in generate method which properly handles
        # ALL 196 image patch tokens (not just mean like before - that was the bug!)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        if "Assistant:" in generated_text:
            generated_text = generated_text.split("Assistant:")[-1].strip()

        return generated_text


def visualize_prediction(
    image_path: str,
    prediction: str,
    output_path: str,
    title: str = None
):
    """Visualize image with prediction text
    
    Args:
        image_path: Path to input image
        prediction: Generated text prediction
        output_path: Path to save visualization
        title: Optional title
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display image
    ax.imshow(image)
    ax.axis('off')
    
    # Add title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Add prediction text below image
    wrapped_text = '\n'.join([prediction[i:i+80] for i in range(0, len(prediction), 80)])
    
    fig.text(
        0.5, 0.02,
        f"Model Prediction:\n{wrapped_text}",
        ha='center',
        va='bottom',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        wrap=True
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to: {output_path}")


def run_inference_on_directory(
    model: MultimodalGemmaInference,
    input_dir: str,
    output_dir: str,
    prompts: List[str] = None
):
    """Run inference on all images in a directory
    
    Args:
        model: Inference model
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        prompts: List of prompts to try (optional)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default prompts
    if prompts is None:
        prompts = [
            "What do you see in this image?",
            "Describe this image in detail.",
            "What is happening in this picture?",
        ]
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in input_path.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    results = []
    for img_file in sorted(image_files):
        logger.info(f"\nProcessing: {img_file.name}")
        
        try:
            # Generate prediction with first prompt
            prompt = prompts[0]
            prediction = model.generate(
                str(img_file),
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.7,
            )
            
            logger.info(f"Prediction: {prediction}")
            
            # Save result
            result = {
                'image': img_file.name,
                'prompt': prompt,
                'prediction': prediction
            }
            results.append(result)
            
            # Create visualization
            output_file = output_path / f"{img_file.stem}_prediction.png"
            visualize_prediction(
                str(img_file),
                prediction,
                str(output_file),
                title=img_file.name
            )
            
        except Exception as e:
            logger.error(f"Error processing {img_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results as text file
    results_file = output_path / "predictions.txt"
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTIMODAL GEMMA INFERENCE RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"Image: {result['image']}\n")
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write("-" * 80 + "\n\n")
    
    logger.info(f"\n✅ Inference complete! Results saved to: {output_dir}")
    logger.info(f"📄 Text results: {results_file}")
    logger.info(f"🖼️  Visualizations: {len(results)} images saved")


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with Multimodal Gemma")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (optional)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing input images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="What do you see in this image?",
        help='Prompt to use for generation'
    )
    
    args = parser.parse_args()
    
    # Initialize model
    model = MultimodalGemmaInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )
    
    # Run inference
    run_inference_on_directory(
        model=model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompts=[args.prompt]
    )


if __name__ == "__main__":
    main()
