import os
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseModel(torch.nn.Module, ABC):
    """Base class for all models."""

    def __init__(self, device: str = "cpu"):
        """
        Initialize the base model.

        Args:
            device: Device to run the model on
        """
        super().__init__()
        self.device = device
        self.to(device)

    def save(self, path: str):
        """
        Save the model to a file.

        Args:
            path: Path to save the model to
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """
        Load the model from a file.

        Args:
            path: Path to load the model from
        """
        self.load_state_dict(torch.load(path, map_location=self.device))


class SimpleNN(BaseModel):
    """Simple neural network for regression tasks."""

    def __init__(self, input_dim: int = 2, hidden_dims: List[int] = [64, 128], device: str = "cpu"):
        """
        Initialize the neural network.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            device: Device to run the model on
        """
        super().__init__(device)

        layers = []
        prev_dim = input_dim

        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim

        # Add output layer
        layers.append(torch.nn.Linear(prev_dim, 1))

        self.layers = torch.nn.ModuleList(layers)
        self.feature_dims = hidden_dims

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Tuple of (output, features) where features is a list of
            intermediate activations
        """
        features = []

        # Track intermediate features for loss prediction
        for i in range(0, len(self.layers) - 1, 2):  # Skip activation layers
            x = self.layers[i](x)  # Linear layer
            x = self.layers[i + 1](x)  # Activation
            features.append(x)

        # Apply the final layer (no activation)
        x = self.layers[-1](x)

        return x, features


class LossNet(BaseModel):
    """Network that predicts the loss of the main model."""

    def __init__(self, feature_dims: List[int] = [64, 128], interm_dim: int = 256, device: str = "cpu"):
        """
        Initialize the loss prediction network.

        Args:
            feature_dims: Dimensions of features from the main model
            interm_dim: Intermediate dimension for feature processing
            device: Device to run the model on
        """
        super().__init__(device)

        # Create feature processing layers
        self.feature_layers = torch.nn.ModuleList([
            torch.nn.Linear(dim, interm_dim)
            for dim in feature_dims
        ])

        # Final layer to predict loss
        self.output_layer = torch.nn.Linear(len(feature_dims) * interm_dim, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to predict loss from features.

        Args:
            features: List of feature tensors from the main model

        Returns:
            Predicted loss values
        """
        processed_features = []

        # Process each feature tensor
        for i, feature in enumerate(features):
            processed = torch.relu(self.feature_layers[i](feature))
            processed_features.append(processed)

        # Concatenate all processed features
        concat_features = torch.cat(processed_features, dim=1)

        # Generate loss prediction
        loss_pred = self.output_layer(concat_features)

        return loss_pred