"""
MushroomANN model definition.

This class is copied verbatim from `mushroom-classification.ipynb` so that the
exported `mushroom_model.pth` state_dict can be loaded outside the notebook.
Architecture: Linear(94→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1) → Sigmoid
"""

from torch import nn


class MushroomANN(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.linear_model_stack = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_model_stack(x)
