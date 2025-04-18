import os
import torch
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Callable, Any

from config.config import ActiveLearningConfig
from models.models import BaseModel, LossNet
from strategies.query_strategies import QueryStrategy

logger = logging.getLogger(__name__)


class ActiveLearner:
    """Core active learning implementation."""

    def __init__(self,
                 config: ActiveLearningConfig,
                 model: torch.nn.Module,
                 query_strategy: QueryStrategy,
                 oracle_fn: Callable[[torch.Tensor], torch.Tensor]):
        """
        Initialize the active learner.

        Args:
            config: Active learning configuration
            model: The model to train
            query_strategy: Strategy to select samples for labeling
            oracle_fn: Function that provides labels for selected samples
        """
        self.config = config
        self.model = model
        self.query_strategy = query_strategy
        self.oracle_fn = oracle_fn

        # Set up device
        self.device = torch.device(config.device)

        # Loss prediction model
        self.loss_model = LossNet(
            feature_dims=model.feature_dims if hasattr(model, 'feature_dims') else [64, 128],
            device=config.device
        )

        # Initialize optimizers
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate
        )
        self.loss_optimizer = torch.optim.SGD(
            self.loss_model.parameters(),
            lr=config.learning_rate
        )

        # Loss function
        self.criterion = torch.nn.MSELoss(reduction='none')

        # Initialize data structures
        self.labeled_indices = []
        self.labeled_inputs = torch.empty((0, config.num_inputs), device=self.device)
        self.labels = torch.empty(0, device=self.device)

        # Performance tracking
        self.losses = []
        self.loss_pred_losses = []

        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def initialize(self, unlabeled_data: torch.Tensor):
        """
        Initialize the active learning process with random samples.

        Args:
            unlabeled_data: Tensor of all available unlabeled data
        """
        unlabeled_indices = torch.randperm(len(unlabeled_data)).tolist()
        initial_indices = unlabeled_indices[:self.config.batch_size]

        # Get labels for initial batch
        initial_inputs = unlabeled_data[initial_indices].to(self.device)
        initial_labels = self.oracle_fn(initial_inputs)

        # Update labeled data
        self.labeled_indices = initial_indices
        self.labeled_inputs = initial_inputs
        self.labels = initial_labels

        logger.info(f"Initialized with {len(initial_indices)} random samples")

    def train_epoch(self, indices=None):
        """
        Train the model for one epoch on the labeled data.

        Args:
            indices: Indices of samples to train on (if None, use all)

        Returns:
            Tuple of (mean_loss, mean_loss_pred_loss)
        """
        self.model.train()
        self.loss_model.train()

        epoch_losses = []
        epoch_loss_pred_losses = []

        # Use all labeled data if no indices provided
        if indices is None:
            if len(self.labeled_indices) <= self.config.epoch_size:
                indices = list(range(len(self.labeled_indices)))
            else:
                # Sample indices based on predicted loss
                with torch.no_grad():
                    outputs, features = self.model(self.labeled_inputs)
                    loss = self.criterion(outputs, self.labels.unsqueeze(1))
                    loss_probs = torch.softmax(loss.squeeze(), dim=0)
                    indices = torch.multinomial(
                        loss_probs,
                        self.config.epoch_size,
                        replacement=False
                    ).tolist()

        # Create batch dataset
        batch_inputs = self.labeled_inputs[indices]
        batch_labels = self.labels[indices].unsqueeze(1)

        # Create DataLoader for batch training
        batch_size = min(self.config.batch_size, len(indices))
        dataset = torch.utils.data.TensorDataset(batch_inputs, batch_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train on batches
        for inputs, targets in loader:
            self.optimizer.zero_grad()
            self.loss_optimizer.zero_grad()

            # Forward pass
            outputs, features = self.model(inputs)
            loss_predictions = self.loss_model(features)

            # Calculate losses
            loss = self.criterion(outputs, targets)
            loss_pred_loss = self.criterion(loss_predictions, loss.detach())

            # Combined loss
            global_loss = loss.mean() + loss_pred_loss.mean()

            # Backward pass
            global_loss.backward()

            # Update weights
            self.optimizer.step()
            self.loss_optimizer.step()

            # Record losses
            epoch_losses.append(loss.mean().item())
            epoch_loss_pred_losses.append(loss_pred_loss.mean().item())

        mean_loss = np.mean(epoch_losses)
        mean_loss_pred_loss = np.mean(epoch_loss_pred_losses)

        return mean_loss, mean_loss_pred_loss

    def query_samples(self, unlabeled_data: torch.Tensor, unlabeled_indices: List[int]):
        """
        Query new samples to label.

        Args:
            unlabeled_data: Tensor of unlabeled data
            unlabeled_indices: Indices of unlabeled samples

        Returns:
            List of selected indices
        """
        # Prepare labeled data tuple if needed by strategy
        labeled_data = None
        if hasattr(self.query_strategy, "needs_labeled_data") and self.query_strategy.needs_labeled_data:
            labeled_data = (self.labeled_inputs, self.labels.unsqueeze(1))

        # Get subset of unlabeled data
        unlabeled_subset = unlabeled_data[unlabeled_indices].to(self.device)

        # Select indices to label
        selected_subset_indices = self.query_strategy.select_samples(
            model=self.model,
            unlabeled_data=unlabeled_subset,
            labeled_data=labeled_data,
            batch_size=self.config.batch_size
        )

        # Map back to original indices
        selected_indices = [unlabeled_indices[i] for i in selected_subset_indices]

        return selected_indices

    def update_labeled_data(self, unlabeled_data: torch.Tensor, selected_indices: List[int]):
        """
        Get labels for selected samples and update labeled data.

        Args:
            unlabeled_data: Tensor of unlabeled data
            selected_indices: Indices of selected samples
        """
        # Get inputs for selected indices
        selected_inputs = unlabeled_data[selected_indices].to(self.device)

        # Get labels from oracle
        selected_labels = self.oracle_fn(selected_inputs)

        # Update labeled data
        self.labeled_inputs = torch.cat([self.labeled_inputs, selected_inputs])
        self.labels = torch.cat([self.labels, selected_labels])
        self.labeled_indices.extend(selected_indices)

    def run(self, unlabeled_data: torch.Tensor):
        """
        Run the active learning process.

        Args:
            unlabeled_data: Tensor of all available unlabeled data

        Returns:
            Trained model and performance metrics
        """
        # Initialize with random samples if needed
        if len(self.labeled_indices) == 0:
            self.initialize(unlabeled_data)

        # Active learning loop
        iteration = 0

        while len(self.labeled_indices) < self.config.max_labels and iteration < self.config.max_iter:
            logger.info(f"Iteration {iteration + 1}/{self.config.max_iter}, "
                        f"Labeled samples: {len(self.labeled_indices)}/{self.config.max_labels}")

            # Train for several epochs
            for epoch in range(self.config.num_epochs_per_iter):
                loss, loss_pred_loss = self.train_epoch()

                self.losses.append(loss)
                self.loss_pred_losses.append(loss_pred_loss)

                logger.info(f"  Epoch {epoch + 1}/{self.config.num_epochs_per_iter}, "
                            f"Loss: {loss:.6f}, Loss Pred Loss: {loss_pred_loss:.6f}")

            # Save models
            self.model.save(os.path.join(self.config.output_dir, f"model_iter{iteration}.pth"))
            self.loss_model.save(os.path.join(self.config.output_dir, f"loss_model_iter{iteration}.pth"))

            # Query new samples if not at limit
            if len(self.labeled_indices) < self.config.max_labels:
                # Get unlabeled indices
                all_indices = set(range(len(unlabeled_data)))
                labeled_set = set(self.labeled_indices)
                unlabeled_indices = list(all_indices - labeled_set)

                # Query new samples
                selected_indices = self.query_samples(unlabeled_data, unlabeled_indices)

                # Update labeled data
                self.update_labeled_data(unlabeled_data, selected_indices)

            iteration += 1

        # Save final models
        self.model.save(os.path.join(self.config.output_dir, "model_final.pth"))
        self.loss_model.save(os.path.join(self.config.output_dir, "loss_model_final.pth"))

        # Save loss history
        np.savetxt(os.path.join(self.config.output_dir, "losses.txt"), self.losses, fmt="%.6f")
        np.savetxt(os.path.join(self.config.output_dir, "losses_pred_loss.txt"),
                   self.loss_pred_losses, fmt="%.6f")

        # Plot and save loss curves
        self._plot_losses()

        return self.model, {"losses": self.losses, "loss_pred_losses": self.loss_pred_losses}

    def _plot_losses(self):
        """Plot and save the loss curves."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))

            iter_range = range(1, len(self.losses) + 1)
            plt.plot(iter_range, self.losses, 'b-', label='Model Loss')
            plt.plot(iter_range, self.loss_pred_losses, 'r-', label='Loss Prediction Loss')

            plt.xlabel('Training Iteration')
            plt.ylabel('Loss')
            plt.title('Training and Loss Prediction Losses')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.savefig(os.path.join(self.config.output_dir, "loss_curves.png"), dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot losses: {e}")