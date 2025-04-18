import numpy as np
import matplotlib.pyplot as plt
import torch  # Add this import
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Optional


def plot_3d_scatter(x, y, z, title="3D Data Scatter Plot", save_path=None):
    """
    Create a 3D scatter plot of data points.

    Args:
        x: X-axis values
        y: Y-axis values
        z: Z-axis values
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=30, alpha=0.7)

    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_zlabel('Output')
    ax.set_title(title)

    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Output Value')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_model_surface(model, input_scaler, output_scaler, grid_size=50,
                       x_range=None, y_range=None, title="Model Prediction Surface",
                       save_path=None):
    """
    Plot the prediction surface of a model.

    Args:
        model: Trained model
        input_scaler: Scaler used for inputs
        output_scaler: Scaler used for outputs
        grid_size: Resolution of the grid
        x_range: Range for x-axis (min, max)
        y_range: Range for y-axis (min, max)
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    model.eval()

    # Determine ranges from scaler if not provided
    if x_range is None or y_range is None:
        # Extract min and max from scaler
        data_min = input_scaler.data_min_
        data_max = input_scaler.data_max_

        if x_range is None:
            x_range = (data_min[0], data_max[0])
        if y_range is None:
            y_range = (data_min[1], data_max[1])

    # Create grid
    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Create input points
    grid_points = np.column_stack([xx.flatten(), yy.flatten()])

    # Scale inputs
    scaled_points = input_scaler.transform(grid_points)
    grid_tensor = torch.tensor(scaled_points, dtype=torch.float32)

    # Generate predictions
    with torch.no_grad():
        predictions, _ = model(grid_tensor)

    # Reshape predictions to grid
    z_pred = predictions.numpy().reshape(grid_size, grid_size)

    # Inverse transform predictions
    grid_pred_original = output_scaler.inverse_transform(z_pred.flatten().reshape(-1, 1))
    z_pred_original = grid_pred_original.reshape(grid_size, grid_size)

    # Plot surface
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(xx, yy, z_pred_original, cmap='viridis', alpha=0.8,
                              linewidth=0, antialiased=True)

    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_zlabel('Predicted Output')
    ax.set_title(title)

    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Predicted Value')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def compare_query_strategies(strategies_results, save_path=None):
    """
    Compare performance of different query strategies.

    Args:
        strategies_results: Dictionary mapping strategy names to loss lists
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))

    markers = ['o', 'v', 's', 'd', '^', 'p', '*']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, (strategy_name, losses) in enumerate(strategies_results.items()):
        iter_range = range(1, len(losses) + 1)
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]

        plt.plot(iter_range, losses, f'-{color}',
                 label=strategy_name.replace('_', ' ').title(),
                 marker=marker, markersize=6, alpha=0.7)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Performance Comparison of Query Strategies')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()