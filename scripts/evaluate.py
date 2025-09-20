#!/usr/bin/env python3
"""
Evaluation script for the trained multimodal model
"""
import torch
import logging
from pathlib import Path
import sys
from PIL import Image
import requests
from io import BytesIO

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import MultimodalGemmaLightning
from src.utils.logging import setup_logging
from src.utils.config import load_config, merge_configs

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for the multimodal model"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        self.checkpoint_path = checkpoint_path
        
        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        else:
            # Try to load default configs
            model_config = load_config("configs/model_config.yaml")
            training_config = load_config("configs/training_config.yaml")
            data_config = load_config("configs/data_config.yaml")
            self.config = merge_configs([model_config, training_config, data_config])
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
    def _load_model(self):
        """Load model from checkpoint"""
        logger.info(f"Loading model from: {self.checkpoint_path}")
        
        model = MultimodalGemmaLightning.load_from_checkpoint(
            self.checkpoint_path,
            config=self.config
        )
        
        return model
    
    def evaluate_text_only(self, prompt: str, max_length: int = 150):
        """Evaluate with text-only input"""
        tokenizer = self.model.model.tokenizer
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def evaluate_with_image(self, prompt: str, image_path: str, max_length: int = 150):
        """Evaluate with image and text input"""
        tokenizer = self.model.model.tokenizer
        vision_processor = self.model.model.vision_processor
        
        # Load and process image
        if image_path.startswith("http"):
            # Download image from URL
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Load local image
            image = Image.open(image_path).convert('RGB')
        
        # Process image
        image_inputs = vision_processor(images=[image], return_tensors="pt")
        
        # Add image token to prompt
        prompt_with_image = f"<image>\n{prompt}"
        
        # Tokenize text
        text_inputs = tokenizer(prompt_with_image, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.model.generate(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                images=image_inputs["pixel_values"],
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][text_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def run_sample_evaluations(self):
        """Run sample evaluations"""
        logger.info("Running sample evaluations...")
        
        # Text-only evaluation
        logger.info("\n=== Text-only Evaluation ===")
        text_prompt = "Human: What is artificial intelligence?\nAssistant:"
        response = self.evaluate_text_only(text_prompt)
        logger.info(f"Prompt: {text_prompt}")
        logger.info(f"Response: {response}")
        
        # Image evaluation (if we have sample images)
        logger.info("\n=== Image + Text Evaluation ===")
        try:
            # Use a sample COCO image
            image_url = "http://images.cocodataset.org/train2017/000000000009.jpg"
            image_prompt = "Human: What do you see in this image?\nAssistant:"
            response = self.evaluate_with_image(image_prompt, image_url)
            logger.info(f"Prompt: {image_prompt}")
            logger.info(f"Response: {response}")
        except Exception as e:
            logger.warning(f"Could not run image evaluation: {e}")


def main():
    """Main evaluation function"""
    setup_logging()
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained multimodal model")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--prompt", type=str, help="Custom prompt to evaluate")
    parser.add_argument("--image", type=str, help="Path to image for multimodal evaluation")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.checkpoint, args.config)
    
    if args.prompt:
        if args.image:
            # Multimodal evaluation
            response = evaluator.evaluate_with_image(args.prompt, args.image)
        else:
            # Text-only evaluation
            response = evaluator.evaluate_text_only(args.prompt)
        
        print(f"Prompt: {args.prompt}")
        print(f"Response: {response}")
    else:
        # Run sample evaluations
        evaluator.run_sample_evaluations()


if __name__ == "__main__":
    main()
