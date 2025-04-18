import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from torch.nn.functional import softmax


class QueryStrategy(ABC):
    """Base class for all query strategies."""

    @abstractmethod
    def select_samples(self,
                       model: torch.nn.Module,
                       unlabeled_data: torch.Tensor,
                       labeled_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                       batch_size: int = 1) -> List[int]:
        """
        Select samples for labeling.

        Args:
            model: The current model
            unlabeled_data: Tensor of unlabeled data points
            labeled_data: Tuple of (inputs, targets) for already labeled data
            batch_size: Number of samples to select

        Returns:
            List of indices to label
        """
        pass


class RandomSampling(QueryStrategy):
    """Randomly select samples for labeling."""

    def select_samples(self,
                       model: torch.nn.Module,
                       unlabeled_data: torch.Tensor,
                       labeled_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                       batch_size: int = 1) -> List[int]:
        """
        Randomly select samples for labeling.

        Args:
            model: The current model (unused)
            unlabeled_data: Tensor of unlabeled data points
            labeled_data: Tuple of (inputs, targets) for already labeled data (unused)
            batch_size: Number of samples to select

        Returns:
            List of indices to label
        """
        num_samples = len(unlabeled_data)
        indices = np.random.choice(num_samples, size=min(batch_size, num_samples), replace=False)
        return indices.tolist()


class UncertaintySampling(QueryStrategy):
    """Select samples based on model uncertainty."""

    def __init__(self, strategy: str = "least_confidence"):
        """
        Initialize uncertainty sampling.

        Args:
            strategy: Uncertainty measure to use. Options:
                     "least_confidence", "margin", or "entropy"
        """
        if strategy not in ["least_confidence", "margin", "entropy"]:
            raise ValueError("Invalid strategy. Choose from 'least_confidence', 'margin', or 'entropy'.")
        self.strategy = strategy

    def select_samples(self,
                       model: torch.nn.Module,
                       unlabeled_data: torch.Tensor,
                       labeled_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                       batch_size: int = 1) -> List[int]:
        """
        Select samples based on model uncertainty.

        Args:
            model: The current model
            unlabeled_data: Tensor of unlabeled data points
            labeled_data: Tuple of (inputs, targets) for already labeled data (unused)
            batch_size: Number of samples to select

        Returns:
            List of indices to label
        """
        model.eval()

        with torch.no_grad():
            outputs, _ = model(unlabeled_data)
            probabilities = softmax(outputs, dim=1).cpu().numpy()

        if self.strategy == "least_confidence":
            scores = 1 - np.max(probabilities, axis=1)
        elif self.strategy == "margin":
            sorted_probs = np.sort(probabilities, axis=1)
            scores = sorted_probs[:, -1] - sorted_probs[:, -2]
        elif self.strategy == "entropy":
            scores = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)

        # Select samples with highest uncertainty scores
        indices = np.argsort(scores)[-batch_size:][::-1]
        return indices.tolist()


class DensityWeightedSampling(QueryStrategy):
    """Select samples based on feature space density."""

    def select_samples(self,
                       model: torch.nn.Module,
                       unlabeled_data: torch.Tensor,
                       labeled_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                       batch_size: int = 1) -> List[int]:
        """
        Select samples based on feature space density.

        Args:
            model: The current model
            unlabeled_data: Tensor of unlabeled data points
            labeled_data: Tuple of (inputs, targets) for already labeled data (unused)
            batch_size: Number of samples to select

        Returns:
            List of indices to label
        """
        model.eval()

        with torch.no_grad():
            _, features = model(unlabeled_data)
            embeddings = features[-1].cpu().numpy()  # Use last layer features

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Calculate density scores (higher means more representative)
        density_scores = np.sum(similarity_matrix, axis=1)

        # Select samples with highest density
        indices = np.argsort(density_scores)[-batch_size:][::-1]
        return indices.tolist()


class QueryByCommittee(QueryStrategy):
    """Select samples based on disagreement among committee members."""

    def select_samples(self,
                       model: torch.nn.Module,
                       unlabeled_data: torch.Tensor,
                       labeled_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                       batch_size: int = 1) -> List[int]:
        """
        Select samples based on committee disagreement.

        Args:
            model: The current model
            unlabeled_data: Tensor of unlabeled data points
            labeled_data: Tuple of (inputs, targets) for already labeled data
            batch_size: Number of samples to select

        Returns:
            List of indices to label
        """
        if labeled_data is None:
            raise ValueError("Labeled data is required for query by committee")

        x_labeled, y_labeled = labeled_data

        # Convert tensors to numpy for sklearn models
        x_labeled_np = x_labeled.cpu().numpy()
        y_labeled_np = y_labeled.cpu().numpy().ravel()  # Flatten for sklearn

        # Create and train committee models
        committee = self._build_committee(x_labeled_np, y_labeled_np)

        # Get predictions from neural network model
        model.eval()
        with torch.no_grad():
            outputs, _ = model(unlabeled_data)
            nn_probs = softmax(outputs, dim=1).cpu().numpy()

        # Get predictions from committee models
        unlabeled_np = unlabeled_data.cpu().numpy()
        committee_preds = []

        for model in committee:
            preds = model.predict(unlabeled_np)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)

            # Apply softmax to make predictions comparable
            preds = softmax(torch.tensor(preds), dim=1).numpy()
            committee_preds.append(preds)

        # All predictions (including neural network)
        all_preds = [nn_probs] + committee_preds
        all_preds = np.array(all_preds)

        # Ensure all predictions have the same shape
        num_classes = nn_probs.shape[1]
        aligned_preds = []

        for preds in all_preds:
            if preds.shape[1] != num_classes:
                expanded_preds = np.zeros((preds.shape[0], num_classes))
                expanded_preds[:, 0] = preds[:, 0]
                aligned_preds.append(expanded_preds)
            else:
                aligned_preds.append(preds)

        aligned_preds = np.stack(aligned_preds, axis=0)

        # Calculate vote entropy based on average predictions
        avg_probs = np.mean(aligned_preds, axis=0)
        vote_entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-12), axis=1)

        # Select samples with highest disagreement
        indices = np.argsort(vote_entropy)[-batch_size:][::-1]
        return indices.tolist()

    def _build_committee(self, x_labeled: np.ndarray, y_labeled: np.ndarray) -> List:
        """
        Build a committee of models using scikit-learn.

        Args:
            x_labeled: Labeled input data
            y_labeled: Corresponding labels

        Returns:
            List of trained models
        """
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        svm_model = SVR(kernel='linear')
        ridge_model = Ridge(alpha=1.0, random_state=42)

        rf_model.fit(x_labeled, y_labeled)
        svm_model.fit(x_labeled, y_labeled)
        ridge_model.fit(x_labeled, y_labeled)

        return [rf_model, svm_model, ridge_model]


class ExpectedErrorReduction(QueryStrategy):
    """Select samples that maximize expected error reduction."""

    def select_samples(self,
                       model: torch.nn.Module,
                       unlabeled_data: torch.Tensor,
                       labeled_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                       batch_size: int = 1) -> List[int]:
        """
        Select samples that maximize expected error reduction.

        Args:
            model: The current model
            unlabeled_data: Tensor of unlabeled data points
            labeled_data: Tuple of (inputs, targets) for already labeled data
            batch_size: Number of samples to select

        Returns:
            List of indices to label
        """
        if labeled_data is None:
            raise ValueError("Labeled data is required for expected error reduction")

        x_labeled, y_labeled = labeled_data

        model.eval()

        # Calculate baseline loss
        with torch.no_grad():
            outputs, _ = model(x_labeled)
            baseline_loss = torch.nn.functional.mse_loss(outputs, y_labeled).item()

        # Calculate expected error reduction for each unlabeled point
        error_reductions = []

        for i in range(len(unlabeled_data)):
            # Create a single example
            x_i = unlabeled_data[i].unsqueeze(0)

            # Generate pseudo-label
            with torch.no_grad():
                y_i, _ = model(x_i)

            # Augment labeled dataset
            augmented_x = torch.cat([x_labeled, x_i])
            augmented_y = torch.cat([y_labeled, y_i])

            # Calculate new loss
            with torch.no_grad():
                outputs, _ = model(augmented_x)
                augmented_loss = torch.nn.functional.mse_loss(outputs, augmented_y).item()

            # Error reduction (higher is better)
            error_reduction = baseline_loss - augmented_loss
            error_reductions.append(error_reduction)

        # Convert to numpy array
        error_reductions = np.array(error_reductions)

        # Select samples with highest expected error reduction
        indices = np.argsort(error_reductions)[-batch_size:][::-1]
        return indices.tolist()


class LossPredictionSampling(QueryStrategy):
    """Select samples based on predicted loss values."""

    def __init__(self, loss_model: torch.nn.Module):
        """
        Initialize loss prediction sampling.

        Args:
            loss_model: Model that predicts loss values
        """
        self.loss_model = loss_model

    def select_samples(self,
                       model: torch.nn.Module,
                       unlabeled_data: torch.Tensor,
                       labeled_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                       batch_size: int = 1) -> List[int]:
        """
        Select samples based on predicted loss values.

        Args:
            model: The current model
            unlabeled_data: Tensor of unlabeled data points
            labeled_data: Tuple of (inputs, targets) for already labeled data (unused)
            batch_size: Number of samples to select

        Returns:
            List of indices to label
        """
        model.eval()
        self.loss_model.eval()

        with torch.no_grad():
            _, features = model(unlabeled_data)
            loss_scores = self.loss_model(features).squeeze()

        # Convert to probabilities for sampling
        loss_probs = torch.softmax(loss_scores, dim=0).cpu().numpy()

        # Sample based on loss probabilities
        indices = np.random.choice(
            len(unlabeled_data),
            size=min(batch_size, len(unlabeled_data)),
            replace=False,
            p=loss_probs
        )

        return indices.tolist()