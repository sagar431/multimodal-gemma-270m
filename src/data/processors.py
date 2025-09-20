"""
Data processors for images and text
"""
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image preprocessing for CLIP vision encoder"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_size = config["data"]["image_size"]
        
        # CLIP normalization values
        self.mean = config["data"]["image_mean"]
        self.std = config["data"]["image_std"]
        
        # Setup transforms
        self.transform = self._setup_transforms()
        
    def _setup_transforms(self):
        """Setup image transformations"""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]
        
        # Add augmentations if enabled
        if self.config["data"]["augmentation"]["enabled"]:
            aug_transforms = []
            
            # Random resized crop
            if self.config["data"]["augmentation"].get("random_resized_crop"):
                scale = self.config["data"]["augmentation"]["random_resized_crop"]
                aug_transforms.append(
                    transforms.RandomResizedCrop(
                        self.image_size, 
                        scale=(scale, 1.0)
                    )
                )
            
            # Color jitter
            if self.config["data"]["augmentation"].get("color_jitter"):
                brightness = self.config["data"]["augmentation"]["color_jitter"]
                aug_transforms.append(
                    transforms.ColorJitter(brightness=brightness)
                )
            
            # Horizontal flip
            if self.config["data"]["augmentation"].get("horizontal_flip"):
                prob = self.config["data"]["augmentation"]["horizontal_flip"]
                aug_transforms.append(
                    transforms.RandomHorizontalFlip(p=prob)
                )
            
            # Insert augmentations before normalization
            transform_list = (
                transform_list[:-2] +  # Resize, ToTensor
                aug_transforms +
                transform_list[-2:]    # Normalize
            )
        
        return transforms.Compose(transform_list)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Process a single image"""
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")
        
        return self.transform(image)
    
    def process_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Process a batch of images"""
        processed = []
        for img in images:
            processed.append(self(img))
        return torch.stack(processed)


class TextProcessor:
    """Text preprocessing for conversations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_length = config["data"]["max_length"]
        
        # Conversation formatting
        conv_config = config["data"]["conversation"]
        self.system_message = conv_config.get("system_message", "")
        self.user_prefix = conv_config.get("user_prefix", "Human: ")
        self.assistant_prefix = conv_config.get("assistant_prefix", "Assistant: ")
        self.turn_separator = conv_config.get("turn_separator", "\n")
        
    def format_conversation(self, conversations: List[Dict[str, str]]) -> str:
        """Format conversation into training text with robust error handling"""
        formatted_parts = []

        # Add system message if present
        if self.system_message:
            formatted_parts.append(self.system_message)

        # Ensure conversations is a valid list
        if not isinstance(conversations, list):
            conversations = []

        # Process conversation turns with error handling
        for turn in conversations:
            try:
                if not isinstance(turn, dict):
                    continue

                role = turn.get("from", "").lower().strip()
                content = turn.get("value", "")

                # Clean and validate content
                if not isinstance(content, str):
                    content = str(content) if content else ""

                content = content.strip()
                if not content:
                    continue

                # Remove problematic characters that might cause issues
                content = content.replace('\x00', '').replace('\n\n\n', '\n\n')

                if role in ["human", "user"]:
                    formatted_parts.append(f"{self.user_prefix}{content}")
                elif role in ["gpt", "assistant", "ai"]:
                    formatted_parts.append(f"{self.assistant_prefix}{content}")
                else:
                    # Default to human if role is unclear
                    formatted_parts.append(f"{self.user_prefix}{content}")

            except Exception as e:
                logger.debug(f"Error processing conversation turn: {e}")
                continue

        # Ensure we have at least some content
        if not formatted_parts:
            return f"{self.user_prefix}What do you see in this image?{self.turn_separator}{self.assistant_prefix}I can see an image."

        return self.turn_separator.join(formatted_parts)
    
    def add_image_token(self, text: str, has_image: bool = True) -> str:
        """Add image token to text if image is present"""
        if has_image:
            image_token = self.config.get("special_tokens", {}).get("image_token", "<image>")
            return f"{image_token}\n{text}"
        return text
    
    def validate_text(self, text: str) -> bool:
        """Validate text meets filtering criteria - more lenient validation"""
        if not isinstance(text, str):
            return False

        # Basic cleanup
        text = text.strip()

        # Check for completely empty content
        if not text:
            return False

        # More lenient length check - just ensure it's not absurdly long or short
        text_length = len(text)
        if text_length < 5:  # Very short
            return False
        if text_length > 2000:  # Very long
            return False

        # Check for basic structure (should have some content)
        if len(text.split()) < 2:  # Less than 2 words
            return False

        return True
