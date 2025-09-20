"""
Projection layers for multimodal fusion
"""
import torch
import torch.nn as nn
from typing import Optional


class VisionProjector(nn.Module):
    """Projects vision features to language model embedding space"""
    
    def __init__(
        self, 
        vision_dim: int, 
        language_dim: int, 
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        hidden_dim = hidden_dim or language_dim
        
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, language_dim),
            nn.LayerNorm(language_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize projection weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch_size, vision_dim]
        Returns:
            projected_features: [batch_size, language_dim]
        """
        return self.projector(vision_features)


class AudioProjector(nn.Module):
    """Projects audio features to language model embedding space"""
    
    def __init__(
        self, 
        audio_dim: int, 
        language_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(audio_dim, language_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(language_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize projection weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: [batch_size, audio_dim]
        Returns:
            projected_features: [batch_size, language_dim]
        """
        return self.projector(audio_features)
