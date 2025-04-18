Setup the environment:
First, create a Python virtual environment and install the required packages:
bashCopy# Create a virtual environment
python -m venv active_learning_env

# Activate the virtual environment
# On Windows:
active_learning_env\Scripts\activate
# On Unix/MacOS:
source active_learning_env/bin/activate

# Install dependencies
pip install -r requirements.txt

Basic usage:
Run the main script with default parameters:
bashCopypython main.py

Using different query strategies:
You can select different query strategies with the --strategy flag:
bashCopy# Use uncertainty sampling
python main.py --strategy uncertainty

# Use random sampling
python main.py --strategy random

# Use density-weighted sampling
python main.py --strategy density

Using a specific configuration file:
You can provide a different configuration file:
bashCopypython main.py --config custom_config.json

Specifying output directory:
You can set where results should be saved:
bashCopypython main.py --output_dir my_results

Using GPU acceleration:
If you have a CUDA-compatible GPU:
bashCopypython main.py --device cuda

Using real data:
If you have a dataset file:
bashCopypython main.py --data_path path/to/your/data.txt


Examples for Different Scenarios

Quick test with synthetic data:
bashCopypython main.py --strategy random --max_iter 5

Full run with real data using uncertainty sampling:
bashCopypython main.py --strategy uncertainty --data_path data/strain_rate_max_data.txt --output_dir results/uncertainty_run

Compare multiple strategies (would require running multiple commands):
bashCopypython main.py --strategy random --output_dir results/random
python main.py --strategy uncertainty --output_dir results/uncertainty
python main.py --strategy density --output_dir results/density


After running the code, you can find the trained models, loss curves, and visualizations in the specified output directory.