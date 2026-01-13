"""
Multimodal Gemma model implementation
"""
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPVisionModel,
    CLIPProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Any, Optional, Tuple
import logging

from .projectors import VisionProjector

logger = logging.getLogger(__name__)


class MultimodalGemma(nn.Module):
    """Multimodal Gemma model with vision and audio capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Initialize tokenizer first
        self._setup_tokenizer()
        
        # Initialize language model
        self._setup_language_model()
        
        # Initialize vision components
        self._setup_vision_components()

        # Initialize projectors
        self._setup_projectors()
        
        # Freeze encoders
        self._freeze_encoders()
        
        # Setup LoRA
        self._setup_lora()
        
        logger.info("MultimodalGemma model initialized successfully")

        # Move projectors to the same device as the language model
        self._move_to_device()
    
    def _setup_tokenizer(self):
        """Initialize and configure tokenizer"""
        model_name = self.config["model"]["gemma_model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Add special tokens
        special_tokens = self.config.get("special_tokens", {})
        new_tokens = []
        
        for token_name, token_value in special_tokens.items():
            if token_value not in self.tokenizer.get_vocab():
                new_tokens.append(token_value)
        
        if new_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            logger.info(f"Added special tokens: {new_tokens}")
    
    def _setup_language_model(self):
        """Initialize language model with quantization if specified"""
        model_name = self.config["model"]["gemma_model_name"]
        
        # Setup quantization config
        quantization_config = None
        if self.config["model"].get("use_4bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config["model"]["bnb_4bit_compute_dtype"]),
                bnb_4bit_quant_type=self.config["model"]["bnb_4bit_quant_type"],
                bnb_4bit_use_double_quant=self.config["model"]["use_nested_quant"]
            )
        
        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map=None,  # Lightning handles device placement
            trust_remote_code=True,
            attn_implementation="eager"  # Use eager attention (flash_attn not required)
        )
        
        # Resize embeddings if we added special tokens
        if len(self.tokenizer) > self.language_model.config.vocab_size:
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized embeddings to {len(self.tokenizer)}")

        # Store image token ID for later use
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(
            self.config.get("special_tokens", {}).get("image_token", "<image>")
        )
    
    def _setup_vision_components(self):
        """Initialize vision encoder and processor"""
        vision_model_name = self.config["model"]["vision_model_name"]
        
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            vision_model_name,
            torch_dtype=torch.bfloat16
        )
        self.vision_processor = CLIPProcessor.from_pretrained(vision_model_name)
        
        logger.info(f"Loaded vision model: {vision_model_name}")
    
    
    def _setup_projectors(self):
        """Initialize projection layers"""
        vision_dim = self.vision_encoder.config.hidden_size
        language_dim = self.language_model.config.hidden_size
        
        # Vision projector
        self.vision_projector = VisionProjector(
            vision_dim=vision_dim,
            language_dim=language_dim,
            hidden_dim=self.config["model"].get("projector_hidden_dim", language_dim)
        ).to(torch.bfloat16)  # Match the model dtype

        logger.info("Initialized vision projection layer")
    
    def _freeze_encoders(self):
        """Freeze vision encoder"""
        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        logger.info("Froze vision encoder parameters")
    
    def _setup_lora(self):
        """Setup LoRA for the language model"""
        lora_config = LoraConfig(
            r=self.config["model"]["lora"]["r"],
            lora_alpha=self.config["model"]["lora"]["alpha"],
            target_modules=self.config["model"]["lora"]["target_modules"],
            lora_dropout=self.config["model"]["lora"]["dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.print_trainable_parameters()
        
        logger.info("Setup LoRA adapters")

    def _move_to_device(self):
        """Move all components to the same device as the language model"""
        device = next(self.language_model.parameters()).device

        # Move vision components
        self.vision_encoder = self.vision_encoder.to(device)
        self.vision_projector = self.vision_projector.to(device)

        logger.info(f"Moved vision components to device: {device}")
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using CLIP and project to language space

        Args:
            images: [batch_size, 3, height, width]
        Returns:
            projected_features: [batch_size, num_image_tokens, language_dim]
        """
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=images)
            # Use ALL patch tokens, not just CLS - this preserves spatial information
            # last_hidden_state shape: [batch_size, num_patches + 1, vision_dim]
            # We skip the CLS token (index 0) and use only patch tokens
            image_features = vision_outputs.last_hidden_state[:, 1:, :]  # [batch, num_patches, 1024]

        # Project to language model space
        # Output: [batch_size, num_patches, language_dim]
        projected_features = self.vision_projector(image_features)
        return projected_features
    
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multimodal inputs

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            images: [batch_size, 3, height, width] or None
            labels: [batch_size, seq_len] or None

        Returns:
            Dictionary with loss and logits
        """
        if images is not None:
            # Encode images and project to language space
            image_features = self.encode_images(images)  # [batch_size, language_dim]

            # Replace <image> tokens with actual image features
            input_embeds, attention_mask, labels = self._merge_image_features(
                input_ids, image_features, attention_mask, labels
            )

            # Forward through language model with merged embeddings
            outputs = self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            # Standard text-only forward pass
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def _merge_image_features(
        self,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge image features with text embeddings by replacing <image> token
        with multiple image patch tokens.

        Args:
            input_ids: [batch_size, seq_len]
            image_features: [batch_size, num_image_tokens, language_dim]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]

        Returns:
            input_embeds: [batch_size, new_seq_len, hidden_size]
            attention_mask: [batch_size, new_seq_len]
            labels: [batch_size, new_seq_len]
        """
        batch_size, seq_len = input_ids.shape
        num_image_tokens = image_features.shape[1]
        device = input_ids.device

        # Get text embeddings
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Find positions of <image> tokens
        image_token_mask = (input_ids == self.image_token_id)

        # Calculate new sequence length (replacing 1 <image> token with num_image_tokens)
        # We assume one <image> token per sample
        new_seq_len = seq_len - 1 + num_image_tokens

        # Create new tensors for the expanded sequence
        new_embeds = torch.zeros(batch_size, new_seq_len, text_embeds.shape[-1],
                                  dtype=text_embeds.dtype, device=device)
        new_attention_mask = torch.zeros(batch_size, new_seq_len,
                                          dtype=attention_mask.dtype, device=device)
        new_labels = torch.full((batch_size, new_seq_len), -100,
                                 dtype=labels.dtype, device=device) if labels is not None else None

        for batch_idx in range(batch_size):
            image_positions = torch.where(image_token_mask[batch_idx])[0]

            if len(image_positions) > 0:
                img_pos = image_positions[0].item()

                # Copy embeddings before <image> token
                new_embeds[batch_idx, :img_pos] = text_embeds[batch_idx, :img_pos]

                # Insert all image tokens
                new_embeds[batch_idx, img_pos:img_pos + num_image_tokens] = image_features[batch_idx]

                # Copy embeddings after <image> token
                remaining_len = seq_len - img_pos - 1
                new_embeds[batch_idx, img_pos + num_image_tokens:img_pos + num_image_tokens + remaining_len] = \
                    text_embeds[batch_idx, img_pos + 1:img_pos + 1 + remaining_len]

                # Handle attention mask
                new_attention_mask[batch_idx, :img_pos] = attention_mask[batch_idx, :img_pos]
                new_attention_mask[batch_idx, img_pos:img_pos + num_image_tokens] = 1  # Attend to image tokens
                new_attention_mask[batch_idx, img_pos + num_image_tokens:img_pos + num_image_tokens + remaining_len] = \
                    attention_mask[batch_idx, img_pos + 1:img_pos + 1 + remaining_len]

                # Handle labels - mask image positions with -100 (ignore in loss)
                if labels is not None:
                    new_labels[batch_idx, :img_pos] = labels[batch_idx, :img_pos]
                    # Image tokens are masked with -100 (already initialized)
                    new_labels[batch_idx, img_pos + num_image_tokens:img_pos + num_image_tokens + remaining_len] = \
                        labels[batch_idx, img_pos + 1:img_pos + 1 + remaining_len]
            else:
                # No image token found - just copy everything
                new_embeds[batch_idx, :seq_len] = text_embeds[batch_idx]
                new_attention_mask[batch_idx, :seq_len] = attention_mask[batch_idx]
                if labels is not None:
                    new_labels[batch_idx, :seq_len] = labels[batch_idx]

        return new_embeds, new_attention_mask, new_labels
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate text with multimodal context"""

        if images is not None:
            # Encode images and merge with text embeddings
            image_features = self.encode_images(images)
            input_embeds, attention_mask, _ = self._merge_image_features(
                input_ids, image_features, attention_mask, None
            )

            # Generate using language model with merged embeddings
            with torch.no_grad():
                outputs = self.language_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
        else:
            # Standard text-only generation
            with torch.no_grad():
                outputs = self.language_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

        return outputs
