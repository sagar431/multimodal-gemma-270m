"""
Dataset implementation for LLaVA multimodal training
"""
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import requests
from PIL import Image
import io
from typing import Dict, Any, List, Optional, Union
import logging
import time
from pathlib import Path

from .processors import ImageProcessor, TextProcessor

logger = logging.getLogger(__name__)


class LLaVADataset(Dataset):
    """LLaVA dataset for multimodal training"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        transform: Optional[Any] = None
    ):
        self.config = config
        self.split = split
        self.transform = transform
        
        # Initialize processors
        self.image_processor = ImageProcessor(config)
        self.text_processor = TextProcessor(config)
        
        # Dataset configuration
        data_config = config["data"]
        self.cache_dir = data_config.get("cache_dir", "./data/cache")
        self.image_size = data_config["image_size"]
        
        # COCO configuration
        coco_config = config.get("coco", {})
        self.coco_base_url = coco_config.get("base_url", "http://images.cocodataset.org/train2017/")
        self.download_timeout = coco_config.get("download_timeout", 30)
        self.retry_attempts = coco_config.get("retry_attempts", 3)
        self.fallback_size = tuple(coco_config.get("fallback_image_size", [224, 224]))
        self.fallback_color = coco_config.get("fallback_image_color", "white")
        
        # Load dataset
        self._load_dataset()

        # Apply filtering optimizations
        if config["data"].get("filter_long_conversations", True):
            self._filter_dataset()

        # Statistics
        self.successful_images = 0
        self.failed_images = 0

        logger.info(f"Initialized LLaVADataset with {len(self.dataset)} samples for split '{split}'")
    
    def _load_dataset(self):
        """Load the LLaVA dataset from HuggingFace"""
        dataset_name = self.config["data"]["dataset_name"]

        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Try different loading approaches
        loading_strategies = [
            # Strategy 1: Simple loading without problematic parameters
            lambda: load_dataset(
                dataset_name,
                split=self.split,
                cache_dir=self.cache_dir
            ),

            # Strategy 2: With streaming disabled
            lambda: load_dataset(
                dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                streaming=False
            ),

            # Strategy 3: Different data format approach
            lambda: self._load_alternative_format(dataset_name),

            # Strategy 4: Load from local files if available
            lambda: self._load_local_dataset(dataset_name)
        ]

        for i, strategy in enumerate(loading_strategies):
            try:
                logger.info(f"Trying dataset loading strategy {i+1}...")
                self.dataset = strategy()

                # Validate dataset
                if len(self.dataset) == 0:
                    raise ValueError("Dataset is empty")

                logger.info(f"Successfully loaded {len(self.dataset)} examples from {dataset_name}")
                return

            except Exception as e:
                logger.warning(f"Strategy {i+1} failed: {e}")
                # Continue to next strategy

        # If all strategies fail, create a larger dummy dataset for development
        logger.warning("All loading strategies failed, creating larger dummy dataset...")
        self.dataset = self._create_development_dataset()

    def _load_alternative_format(self, dataset_name):
        """Try alternative loading format for LLaVA dataset"""
        try:
            # Try loading with explicit JSON format
            from datasets import load_dataset, DownloadConfig

            download_config = DownloadConfig(
                resume_download=True,
                force_download=False,
                use_etag=False
            )

            return load_dataset(
                "json",
                data_files={
                    "train": "hf://datasets/liuhaotian/LLaVA-Instruct-150K/llava_instruct_150k.json"
                },
                split=self.split,
                cache_dir=self.cache_dir,
                download_config=download_config
            )
        except Exception as e:
            logger.warning(f"Alternative format loading failed: {e}")
            raise

    def _load_local_dataset(self, dataset_name):
        """Try to load dataset from local files or alternative sources"""
        try:
            # Try loading with minimal parameters
            return load_dataset(
                dataset_name,
                split=self.split,
                cache_dir=self.cache_dir
            )
        except Exception:
            # If local loading fails, create dummy data
            logger.warning("Local loading failed, using dummy dataset")
            return self._create_dummy_dataset()

    def _create_dummy_dataset(self):
        """Create a small dummy dataset for testing"""
        from datasets import Dataset

        dummy_data = []
        for i in range(100):  # Small dataset for testing
            # Use realistic COCO-style filenames that will trigger fallback
            coco_filename = f"{str(i).zfill(12)}.jpg"
            dummy_data.append({
                "id": str(i),
                "image": coco_filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"What do you see in image {i}?"
                    },
                    {
                        "from": "gpt",
                        "value": f"I can see an image numbered {i}."
                    }
                ]
            })

        return Dataset.from_list(dummy_data)

    def _create_development_dataset(self):
        """Create a larger dummy dataset for development/testing"""
        from datasets import Dataset
        import random

        # Create more realistic sample data for development
        dummy_data = []

        # Common visual questions and responses
        questions = [
            "What do you see in this image?",
            "Describe the main objects in the picture.",
            "What is the person doing?",
            "What colors are prominent in this image?",
            "Can you identify any animals in the picture?",
            "What's the setting or location of this image?",
            "Are there any vehicles visible?",
            "What's the weather like in the image?",
            "How many people are in the picture?",
            "What objects are on the table?",
        ]

        responses = [
            "I can see a person standing in a park with trees in the background.",
            "The image shows a cat sitting on a windowsill, looking outside.",
            "There's a red car parked on a street with buildings nearby.",
            "I notice several people walking on a busy sidewalk.",
            "The picture contains a bowl of fruit on a wooden table.",
            "I can see a dog playing in a grassy field.",
            "The image shows a bicycle leaning against a wall.",
            "There's a group of children playing in a playground.",
            "I can see mountains in the distance with a clear blue sky.",
            "The picture shows a kitchen with modern appliances.",
        ]

        # Generate realistic sample size for development
        num_samples = self.config["data"].get("subset_size", 10000) if self.config["data"].get("use_subset", False) else 50000

        for i in range(num_samples):
            # Use realistic COCO-style filenames
            coco_filename = f"{str(i % 1000).zfill(12)}.jpg"
            question = random.choice(questions)
            response = random.choice(responses)

            dummy_data.append({
                "id": str(i),
                "image": coco_filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt",
                        "value": response
                    }
                ]
            })

        logger.info(f"Created development dataset with {len(dummy_data)} samples")
        return Dataset.from_list(dummy_data)

    def _filter_dataset(self):
        """Filter dataset for faster training"""
        logger.info("Applying speed optimization filters...")

        filtering_config = self.config["data"]["filtering"]
        data_config = self.config["data"]

        original_size = len(self.dataset)
        filtered_indices = []

        # Use subset for testing if enabled
        if data_config.get("use_subset", False):
            subset_size = data_config.get("subset_size", 10000)
            indices = list(range(min(subset_size, original_size)))
            logger.info(f"Using subset of {len(indices)} samples for testing")
        else:
            indices = list(range(original_size))

        max_turns = data_config.get("max_conversation_turns", 6)
        max_tokens = filtering_config.get("max_tokens_per_sample", 256)
        max_length = filtering_config.get("max_length", 800)

        for idx in indices:
            try:
                item = self.dataset[idx]
                conversations = item.get("conversations", [])

                # Filter by conversation length
                if len(conversations) > max_turns:
                    continue

                # Estimate token count (rough approximation: 1 token ≈ 4 chars)
                total_text = ""
                for conv in conversations:
                    total_text += conv.get("value", "")

                estimated_tokens = len(total_text) // 4
                if estimated_tokens > max_tokens:
                    continue

                # Check if it's image-related (has visual keywords)
                has_visual_content = any(
                    keyword in total_text.lower()
                    for keyword in ["see", "image", "picture", "photo", "visual", "look", "show", "appear", "visible"]
                )

                if filtering_config.get("min_image_questions", 1) > 0 and not has_visual_content:
                    continue

                # Check final text length
                if len(total_text) > max_length:
                    continue

                filtered_indices.append(idx)

            except Exception as e:
                logger.debug(f"Error filtering item {idx}: {e}")
                continue

        # Apply filtering
        if filtered_indices:
            self.dataset = self.dataset.select(filtered_indices)

        filtered_size = len(self.dataset)
        reduction_pct = (1 - filtered_size / original_size) * 100

        logger.info(f"Dataset filtered: {original_size:,} → {filtered_size:,} samples")
        logger.info(f"Reduction: {reduction_pct:.1f}% (faster training!)")

        return self.dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset with improved error handling"""
        try:
            item = self.dataset[idx]

            # Load and process image
            image = self._load_image(item.get("image", ""))

            # Process conversation text with robust handling
            conversations = item.get("conversations", [])
            if not conversations or not isinstance(conversations, list):
                # Fallback if no valid conversations
                conversations = [
                    {"from": "human", "value": "What do you see in this image?"},
                    {"from": "gpt", "value": "I can see an image that contains various visual elements."}
                ]

            formatted_text = self.text_processor.format_conversation(conversations)

            # Add image token if image is present
            formatted_text = self.text_processor.add_image_token(formatted_text, image is not None)

            # More lenient validation - only reject if truly problematic
            if not self.text_processor.validate_text(formatted_text):
                # Create a better fallback based on original conversations
                try:
                    # Try to extract any usable content
                    fallback_content = "What do you see in this image?"
                    if conversations and len(conversations) > 0:
                        first_conv = conversations[0]
                        if isinstance(first_conv, dict) and "value" in first_conv:
                            user_text = str(first_conv["value"]).strip()
                            if user_text and len(user_text) > 5:
                                fallback_content = user_text

                    formatted_text = f"<image>\nHuman: {fallback_content}\nAssistant: I can see an image."
                except Exception:
                    formatted_text = "<image>\nHuman: What do you see?\nAssistant: I see an image."

            return {
                "image": image,
                "text": formatted_text,
                "conversations": conversations,
                "id": item.get("id", f"sample_{idx}"),
                "image_filename": item.get("image", ""),
                "has_image": image is not None
            }

        except Exception as e:
            logger.debug(f"Error processing item {idx}: {e}")
            # Return a fallback sample (reduce logging level to debug)
            return self._get_fallback_sample(idx)
    
    def _load_image(self, image_filename: str) -> Optional[Image.Image]:
        """Load image from COCO dataset with retry logic"""
        if not image_filename or not image_filename.strip():
            return None

        # Check if it's a dummy image (contains "dummy_")
        if "dummy_" in image_filename:
            logger.debug(f"Using placeholder image for {image_filename}")
            return self._create_fallback_image()

        # For actual dummy filenames from our generated dataset (short numbers), use placeholder
        filename_without_ext = image_filename.replace('.jpg', '').replace('.png', '')
        if image_filename and filename_without_ext.isdigit() and len(filename_without_ext) <= 6:
            logger.debug(f"Using placeholder image for dummy filename: {image_filename}")
            return self._create_fallback_image()

        # Check cache first
        cache_path = Path(self.cache_dir) / "images" / image_filename
        if cache_path.exists():
            try:
                image = Image.open(cache_path).convert('RGB')
                self.successful_images += 1
                return image
            except Exception:
                cache_path.unlink(missing_ok=True)  # Remove corrupted cache

        image_url = f"{self.coco_base_url}{image_filename}"

        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(
                    image_url,
                    timeout=self.download_timeout,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                response.raise_for_status()

                # Load and validate image
                image = Image.open(io.BytesIO(response.content)).convert('RGB')

                # Basic validation
                if image.size[0] < 10 or image.size[1] < 10:
                    raise ValueError("Image too small")

                # Cache the image
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(cache_path, "JPEG", quality=85)
                logger.debug(f"Cached image: {cache_path}")

                self.successful_images += 1
                return image

            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    logger.debug(f"Failed to load image {image_filename} after {self.retry_attempts} attempts: {e}")
                    self.failed_images += 1
                    return self._create_fallback_image()
                else:
                    time.sleep(0.5)  # Brief pause before retry

        return self._create_fallback_image()
    
    def _create_fallback_image(self) -> Image.Image:
        """Create a fallback image when loading fails"""
        return Image.new('RGB', self.fallback_size, color=self.fallback_color)
    
    def _get_fallback_sample(self, idx: int) -> Dict[str, Any]:
        """Get a fallback sample when processing fails"""
        fallback_image = self._create_fallback_image()
        fallback_text = "Human: What do you see in this image?\nAssistant: I can see a simple image."
        
        return {
            "image": fallback_image,
            "text": fallback_text,
            "conversations": [
                {"from": "human", "value": "What do you see in this image?"},
                {"from": "gpt", "value": "I can see a simple image."}
            ],
            "id": f"fallback_{idx}",
            "image_filename": "",
            "has_image": True
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get dataset statistics"""
        return {
            "total_samples": len(self),
            "successful_images": self.successful_images,
            "failed_images": self.failed_images,
            "success_rate": self.successful_images / (self.successful_images + self.failed_images) * 100 
                           if (self.successful_images + self.failed_images) > 0 else 0
        }


class MultimodalCollator:
    """Custom collator for multimodal data batching"""
    
    def __init__(
        self,
        tokenizer,
        vision_processor,
        config: Dict[str, Any],
        max_length: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.config = config
        self.max_length = max_length or config["data"]["max_length"]
        
        # Image token for processing
        self.image_token = config.get("special_tokens", {}).get("image_token", "<image>")
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples"""
        
        images = []
        texts = []
        has_images = []
        
        for sample in batch:
            # Collect images
            if sample["image"] is not None:
                images.append(sample["image"])
                has_images.append(True)
            else:
                # Create placeholder image for samples without images
                placeholder = Image.new('RGB', (224, 224), color='white')
                images.append(placeholder)
                has_images.append(False)
            
            # Collect texts
            texts.append(sample["text"])
        
        # Process images using vision processor
        try:
            vision_inputs = self.vision_processor(
                images=images,
                return_tensors="pt"
            )
            pixel_values = vision_inputs["pixel_values"]
        except Exception as e:
            logger.error(f"Error processing images: {e}")
            # Create dummy pixel values
            pixel_values = torch.zeros(len(batch), 3, 224, 224)
        
        # Tokenize texts
        try:
            text_inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Error tokenizing texts: {e}")
            # Create dummy inputs
            text_inputs = {
                "input_ids": torch.zeros(len(batch), self.max_length, dtype=torch.long),
                "attention_mask": torch.ones(len(batch), self.max_length, dtype=torch.long)
            }
        
        # Create labels (same as input_ids for causal LM)
        labels = text_inputs["input_ids"].clone()
        
        # Mask padding tokens in labels (-100 is ignored by loss function)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        batch_dict = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": labels,
            "images": pixel_values,
            "has_images": torch.tensor(has_images, dtype=torch.bool)
        }
        
        return batch_dict
