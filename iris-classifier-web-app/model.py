"""
IrisClassifier model definition.
Architecture: Linear(4→16) → ReLU → Linear(16→16) → ReLU → Linear(16→3)
"""

import torch
import torch.nn as nn


class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_model_Stack = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.liner_model_Stack(x)
