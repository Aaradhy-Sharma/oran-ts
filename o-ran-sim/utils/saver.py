import json
import os
from datetime import datetime
import matplotlib.pyplot as plt # Import for saving figures
import numpy as np # For handling NaN values during JSON serialization

class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy types (e.g., np.float64, np.int64, np.nan).
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                # JSON doesn't have NaN/Infinity, convert to string or null.
                # String "NaN", "Infinity", "-Infinity" is a common convention.
                return str(obj)
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class SaveHandler:
    """
    Handles saving simulation metrics to a JSON file and plots to image files.
    """
    def __init__(self, output_dir="simulation_results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_metrics(self, agent_name: str, metrics_history: list, sim_params: dict, experiment_name: str = None):
        """
        Saves simulation metrics and parameters to a JSON file.

        Args:
            agent_name (str): Name of the RL agent used for this run.
            metrics_history (list): List of dictionaries, each containing step metrics.
            sim_params (dict): Dictionary of simulation parameters for this run.
            experiment_name (str, optional): Name of the experiment for subdirectory.
        """
        sub_dir = os.path.join(self.output_dir, experiment_name) if experiment_name else self.output_dir
        os.makedirs(sub_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{agent_name}_metrics_{timestamp}.json"
        filepath = os.path.join(sub_dir, filename)

        data_to_save = {
            "agent_name": agent_name,
            "timestamp": timestamp,
            "simulation_parameters": sim_params,
            "metrics_history": metrics_history,
        }

        try:
            with open(filepath, 'w') as f:
                # Use custom encoder to handle NumPy types, especially NaN/Infinity
                json.dump(data_to_save, f, indent=4, cls=NpEncoder)
            print(f"Metrics saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving metrics to {filepath}: {e}")
            return None

    def save_figure(self, fig: plt.Figure, filename: str, experiment_name: str = None, dpi=300):
        """
        Saves a matplotlib figure to an image file (e.g., .png).

        Args:
            fig (plt.Figure): The matplotlib Figure object to save.
            filename (str): The desired filename (e.g., "throughput_comparison.png").
            experiment_name (str, optional): Name of the experiment for subdirectory.
            dpi (int): Dots per inch for image resolution.
        """
        sub_dir = os.path.join(self.output_dir, experiment_name) if experiment_name else self.output_dir
        os.makedirs(sub_dir, exist_ok=True)

        filepath = os.path.join(sub_dir, filename)
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving figure to {filepath}: {e}")
            return None