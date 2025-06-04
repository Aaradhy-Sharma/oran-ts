import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
                return str(obj) # Or None, depends on desired JSON representation
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

    def save_to_csv(self, data: dict, filename: str, experiment_name: str = None):
        """
        Saves aggregated metrics data to a CSV file.

        Args:
            data (dict): A dictionary where keys are agent names and values are lists of metric dictionaries.
                         Example: {'AgentA': [{...}, {...}], 'AgentB': [{...}, {...}]}
            filename (str): The desired filename (e.g., "scenario_metrics.csv").
            experiment_name (str, optional): Name of the experiment for subdirectory.
        """
        sub_dir = os.path.join(self.output_dir, experiment_name) if experiment_name else self.output_dir
        os.makedirs(sub_dir, exist_ok=True)

        filepath = os.path.join(sub_dir, filename)

        # Prepare data for DataFrame: A list of dictionaries, where each dict is a row
        # We need to flatten the structure and add 'Agent' and 'Scenario' columns
        csv_rows = []
        for agent_name, metrics_history in data.items():
            for step_metrics in metrics_history:
                row = step_metrics.copy()
                row['Agent'] = agent_name
                # 'Scenario' column will be added in runner, as `experiment_name` is the scenario name
                csv_rows.append(row)
        
        if not csv_rows:
            print(f"No data to save to CSV for {filename}.")
            return None

        try:
            df = pd.DataFrame(csv_rows)
            # Reorder columns to put Agent and Time_Step first
            cols = ['Agent', 'time_step'] + [col for col in df.columns if col not in ['Agent', 'time_step']]
            df = df[cols]
            
            df.to_csv(filepath, index=False)
            print(f"CSV data saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving CSV to {filepath}: {e}")
            return None