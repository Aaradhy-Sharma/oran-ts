# utils/saver.py

import json
import os
from datetime import datetime

class SaveHandler:
    """
    Handles saving simulation metrics to a JSON file.
    """
    def __init__(self, output_dir="simulation_results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_metrics(self, agent_name: str, metrics_history: list, sim_params: dict):
        """
        Saves simulation metrics and parameters to a JSON file.

        Args:
            agent_name (str): Name of the RL agent used for this run.
            metrics_history (list): List of dictionaries, each containing step metrics.
            sim_params (dict): Dictionary of simulation parameters for this run.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{agent_name}_metrics_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        data_to_save = {
            "agent_name": agent_name,
            "timestamp": timestamp,
            "simulation_parameters": sim_params,
            "metrics_history": metrics_history,
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print(f"Metrics saved to: {filepath}")
        except Exception as e:
            print(f"Error saving metrics to {filepath}: {e}")

    # You could add methods to load metrics later if needed for persistent comparison
    # def load_metrics(self, filepath: str):
    #     try:
    #         with open(filepath, 'r') as f:
    #             data = json.load(f)
    #         return data
    #     except Exception as e:
    #         print(f"Error loading metrics from {filepath}: {e}")
    #         return None