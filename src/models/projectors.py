"""
Projection layers for multimodal fusion
"""
import torch
import torch.nn as nn
from typing import Optional


class VisionProjector(nn.Module):
    """Projects vision features to language model embedding space.

    Uses a 2-layer MLP with GELU activation following the LLaVA architecture.
    Supports both single vectors and sequences of patch tokens.
    """

    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0  # Reduced dropout for better gradient flow
    ):
        super().__init__()
        hidden_dim = hidden_dim or (language_dim * 2)  # Larger hidden dim for better capacity

        # Two-layer MLP projector (LLaVA-style)
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, language_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights with Xavier for better gradient flow
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights using Xavier initialization"""
        for module in [self.fc1, self.fc2]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to language model embedding space.

        Args:
            vision_features: [batch_size, num_patches, vision_dim] or [batch_size, vision_dim]
        Returns:
            projected_features: [batch_size, num_patches, language_dim] or [batch_size, language_dim]
        """
        x = self.fc1(vision_features)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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
