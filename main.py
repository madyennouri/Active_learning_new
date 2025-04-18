import os
import logging
import torch
import numpy as np
import argparse

from config.config import ActiveLearningConfig
from data.data_manager import DataManager
from models.models import SimpleNN
from strategies.query_strategies import (
    RandomSampling,
    UncertaintySampling,
    DensityWeightedSampling,
    QueryByCommittee,
    ExpectedErrorReduction,
    LossPredictionSampling
)
from core.active_learner import ActiveLearner
from utils.visualization import plot_3d_scatter, plot_model_surface, compare_query_strategies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Active Learning Framework")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to configuration JSON file")
    parser.add_argument("--strategy", type=str, default="uncertainty",
                        choices=["random", "uncertainty", "density", "committee", "error_reduction", "loss_prediction"],
                        help="Query strategy to use")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu, cuda)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to data file (if any)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output files")
    return parser.parse_args()


def main():
    """Main function to run the active learning framework."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = ActiveLearningConfig.from_json(args.config)

    # Override config with command line arguments if provided
    if args.device:
        config.device = args.device
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir

    # Set up data manager
    data_manager = DataManager(config)

    # Prepare data
    if config.data_path:
        # Real data with train/test split
        (x_train, y_train), (x_test, y_test) = data_manager.prepare_data()
        unlabeled_data = x_train

        # Define oracle function using real labels
        def oracle_fn(selected_inputs):
            # Find closest training examples and return their labels
            distances = torch.cdist(selected_inputs, x_train)
            indices = torch.argmin(distances, dim=1)
            return y_train[indices].squeeze()

    else:
        # Synthetic data generation
        unlabeled_data = data_manager.generate_synthetic_inputs()

        # Define synthetic oracle function
        def oracle_fn(selected_inputs):
            # Unscale the inputs
            inputs_np = selected_inputs.cpu().numpy()
            original_inputs = data_manager.input_scaler.inverse_transform(inputs_np)

            # Compute synthetic function output
            x, y = original_inputs[:, 0], original_inputs[:, 1]
            z = np.sin(x ** 2) * np.cos(y ** 2) + np.exp(-((x - 1) ** 2 + (y + 2) ** 2)) * np.sin(3 * x) * np.cos(
                3 * y) + 0.5 * x * y

            # Scale the outputs
            z_scaled = data_manager.output_scaler.fit_transform(z.reshape(-1, 1))
            return torch.tensor(z_scaled.flatten(), device=config.device)

    # Initialize model
    model = SimpleNN(input_dim=config.num_inputs, device=config.device)

    # Choose query strategy based on argument
    if args.strategy == "random":
        query_strategy = RandomSampling()
    elif args.strategy == "uncertainty":
        query_strategy = UncertaintySampling(strategy="entropy")
    elif args.strategy == "density":
        query_strategy = DensityWeightedSampling()
    elif args.strategy == "committee":
        query_strategy = QueryByCommittee()
    elif args.strategy == "error_reduction":
        query_strategy = ExpectedErrorReduction()
    elif args.strategy == "loss_prediction":
        # We'll initialize this later after creating the loss model
        loss_model = None  # Placeholder
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        return

    # Create active learner
    learner = ActiveLearner(
        config=config,
        model=model,
        query_strategy=query_strategy if args.strategy != "loss_prediction" else None,
        oracle_fn=oracle_fn
    )

    # If using loss prediction strategy, set it now
    if args.strategy == "loss_prediction":
        learner.query_strategy = LossPredictionSampling(learner.loss_model)

    # Run active learning
    trained_model, metrics = learner.run(unlabeled_data)

    logger.info("Active learning completed!")
    logger.info(f"Final model saved to {os.path.join(config.output_dir, 'model_final.pth')}")

    # Visualize results if real data is available
    if config.data_path:
        plot_3d_scatter(
            x_test[:, 0].numpy(),
            x_test[:, 1].numpy(),
            y_test.numpy(),
            title="Test Data Distribution",
            save_path=os.path.join(config.output_dir, "test_data.png")
        )

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs, _ = model(x_test)
            test_loss = torch.nn.functional.mse_loss(test_outputs, y_test)

        logger.info(f"Test MSE: {test_loss.item():.6f}")

    # Visualize model prediction surface
    plot_model_surface(
        model=model,
        input_scaler=data_manager.input_scaler,
        output_scaler=data_manager.output_scaler,
        title="Final Model Prediction Surface",
        save_path=os.path.join(config.output_dir, "model_surface.png")
    )


if __name__ == "__main__":
    main()