# model_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# --- Configuration ---
# We can make the model deeper with residual blocks
NUM_INPUT_PLANES = 19 # 12 piece planes + 7 feature planes
NUM_RESIDUAL_BLOCKS = 8
NUM_CHANNELS = 128

class ResidualBlock(nn.Module):
    """
    A standard residual block to allow for deeper networks.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ChessModelV2(nn.Module):
    """
    An upgraded version of your ChessModel with a dual-head (policy/value) design
    and residual blocks, inspired by AlphaZero.
    """
    def __init__(self, num_classes: int, num_blocks: int = NUM_RESIDUAL_BLOCKS, num_channels: int = NUM_CHANNELS):
        super(ChessModelV2, self).__init__()
        
        # Initial convolutional layer
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_INPUT_PLANES, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # A stack of residual blocks
        self.body = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        # --- Policy Head ---
        # Predicts the probability distribution over all possible moves.
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, num_classes)
        )

        # --- Value Head ---
        # Predicts the expected outcome of the game from the current position [-1, 1].
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() # Tanh activation to scale the output to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, NUM_INPUT_PLANES, 8, 8).

        Returns:
            A tuple containing:
            - policy_logits (torch.Tensor): Raw output for the policy head.
            - value (torch.Tensor): The evaluation of the position.
        """
        x = self.stem(x)
        x = self.body(x)
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value