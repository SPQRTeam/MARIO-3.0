"""ResNet18-based model for robot color classification."""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple



class ColorClassifierCNNResnet(nn.Module):
    """
    ResNet18 pretrained on ImageNet, with the final FC layer replaced
    to classify robot jersey colors.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)



class ColorClassifierCNNv0(nn.Module):
    """
    Convolutional Neural Network for robot color classification.

    Architecture:
    - 3 convolutional layers with batch normalization
    - Max pooling after each conv layer
    - 2 fully connected layers with dropout
    - Designed for 28x28 RGB input images
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 7):
        """
        Initialize the CNN model.

        Args:
            in_channels: Number of input channels (3 for RGB).
            num_classes: Number of output classes (robot colors).
        """
        super(ColorClassifierCNNv0, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        # After 3 pooling layers on 28x28 input: 28 -> 14 -> 7 -> 3
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_feature_dims(self) -> Tuple[int, ...]:
        """
        Get the dimensions of the feature map before FC layers.

        Returns:
            Tuple of (channels, height, width) after all conv layers.
        """
        return (128, 3, 3)

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"ColorClassifierCNNv0(\n"
            f"  in_channels={self.in_channels},\n"
            f"  num_classes={self.num_classes},\n"
            f"  parameters={sum(p.numel() for p in self.parameters()):,}\n"
            f")"
        )

