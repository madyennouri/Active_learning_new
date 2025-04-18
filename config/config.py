import json
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningConfig:
    """Configuration parameters for active learning process."""
    num_inputs: int
    resolution: int
    input_max: List[float]
    input_min: List[float]
    batch_size: int
    epoch_size: int
    max_iter: int
    max_labels: int
    num_epochs_per_iter: int
    learning_rate: float = 0.01
    device: str = "cpu"
    random_seed: int = 42
    data_path: Optional[str] = None
    output_dir: str = "results"

    @classmethod
    def from_json(cls, file_path: str) -> 'ActiveLearningConfig':
        """Load configuration from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
                # Set default values for optional fields if they don't exist
                if 'learning_rate' not in config_dict:
                    config_dict['learning_rate'] = 0.01
                if 'device' not in config_dict:
                    config_dict['device'] = "cpu"
                if 'random_seed' not in config_dict:
                    config_dict['random_seed'] = 42
                if 'data_path' not in config_dict:
                    config_dict['data_path'] = None
                if 'output_dir' not in config_dict:
                    config_dict['output_dir'] = "results"
                return cls(**config_dict)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configuration: {e}")
            raise