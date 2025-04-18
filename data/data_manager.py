import numpy as np
import torch
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from config.config import ActiveLearningConfig


class IndexedDataset(torch.utils.data.Dataset):
    """Dataset that returns indices alongside data for tracking selected samples."""

    def __init__(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Initialize the dataset.

        Args:
            inputs: Input tensor data
            targets: Target tensor data (optional for unlabeled data)
        """
        self.inputs = inputs
        self.targets = targets
        self.has_targets = targets is not None

    def __getitem__(self, index: int) -> Tuple:
        """
        Get an item by index.

        Returns:
            A tuple of (input, target, index) if targets exist, else (input, index)
        """
        if self.has_targets:
            return self.inputs[index], self.targets[index], index
        else:
            return self.inputs[index], index

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.inputs)


class DataManager:
    """Handles data loading, preprocessing, and splitting."""

    def __init__(self, config: ActiveLearningConfig):
        """
        Initialize the data manager.

        Args:
            config: Active learning configuration
        """
        self.config = config
        self.input_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()

    def load_data_from_file(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load data from a text file.

        Args:
            file_path: Path to the data file

        Returns:
            Tuple of (inputs, outputs) tensors
        """
        try:
            data = np.loadtxt(file_path)
            inputs = torch.tensor(data[:, :2], dtype=torch.float32)
            outputs = torch.tensor(data[:, 2], dtype=torch.float32).reshape(-1, 1)
            return inputs, outputs
        except Exception as e:
            raise Exception(f"Error loading data from {file_path}: {e}")

    def generate_synthetic_inputs(self) -> torch.Tensor:
        """
        Generate a grid of synthetic input points.

        Returns:
            Tensor of input points
        """
        # Create an array of ranges for each dimension
        ranges = [np.linspace(min_val, max_val, self.config.resolution)
                  for min_val, max_val in zip(self.config.input_min, self.config.input_max)]

        # Create the meshgrid
        grids = np.meshgrid(*ranges, indexing='ij')

        # Stack and reshape to create the input matrix
        input_matrix = np.column_stack([grid.ravel() for grid in grids])

        # Scale and convert to tensor
        scaled_inputs = self.input_scaler.fit_transform(input_matrix)
        return torch.tensor(scaled_inputs, dtype=torch.float32)

    def prepare_data(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare data based on configuration.

        Returns:
            Tuple of (inputs, outputs) where outputs may be None for synthetic data
        """
        if self.config.data_path:
            inputs, outputs = self.load_data_from_file(self.config.data_path)

            # Scale the data
            inputs_np = inputs.numpy()
            outputs_np = outputs.numpy()

            # Split for validation
            x_train, x_test, y_train, y_test = train_test_split(
                inputs_np, outputs_np,
                test_size=0.2,
                random_state=self.config.random_seed
            )

            # Scale the data
            x_train_scaled = self.input_scaler.fit_transform(x_train)
            y_train_scaled = self.output_scaler.fit_transform(y_train)

            x_test_scaled = self.input_scaler.transform(x_test)
            y_test_scaled = self.output_scaler.transform(y_test)

            # Convert back to tensors
            x_train = torch.tensor(x_train_scaled, dtype=torch.float32)
            y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
            x_test = torch.tensor(x_test_scaled, dtype=torch.float32)
            y_test = torch.tensor(y_test_scaled, dtype=torch.float32)

            return (x_train, y_train), (x_test, y_test)
        else:
            # Generate synthetic inputs only
            inputs = self.generate_synthetic_inputs()
            return inputs, None