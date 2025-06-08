import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import collections
import os
from tqdm import tqdm
import sys
from contextlib import redirect_stdout
import argparse

# Import core simulation components
from sim_core.params import SimParams
from sim_core.simulation import Simulation
from sim_core.constants import TF_AVAILABLE as TF_AVAILABLE_GLOBAL

# Import utilities
from utils.logger import LogHandler
from utils.saver import SaveHandler

def setup_logging():
    """Setup logging to file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "simulation_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"simulation_{timestamp}.log")
    return log_file

def run_single_simulation(agent_type: str, params: SimParams, run_seed: int, log_file: str):
    """Runs a single simulation with a specific agent type."""
    with open(log_file, 'a') as f:
        with redirect_stdout(f):
            print(f"\n--- Running: {agent_type} (Seed: {run_seed}) ---")
            print(f"Placement Method: {params.placement_method}")
            if params.placement_method == "PPP":
                print(f"Lambda BS: {params.lambda_bs}, Lambda UE: {params.lambda_ue}")

            # Set random seeds
            random.seed(run_seed)
            np.random.seed(run_seed)
            if TF_AVAILABLE_GLOBAL:
                import tensorflow as tf
                tf.random.set_seed(run_seed)

            # Set the RL agent type
            params.rl_agent_type = agent_type

            try:
                simulation = Simulation(params, print)  # Using print as logger
                print(f"Simulation setup for {agent_type} complete. Starting steps...")

                for step in range(params.total_sim_steps):
                    if not simulation.run_step():
                        print(f"Simulation for {agent_type} ended early at step {step}.")
                        break

                print(f"Simulation for {agent_type} completed {simulation.current_time_step} steps.")
                return simulation.metrics_history

            except Exception as e:
                print(f"ERROR: Simulation for {agent_type} failed: {e}")
                return None

def create_research_plot(metric_key, plot_title, all_metrics, save_handler, exp_folder_name):
    """Creates a single research-quality plot for a specific metric."""
    plt.style.use('seaborn-v0_8-paper')  # Use a clean, research-friendly style
    
    # Create figure with specific size for research papers
    fig, ax = plt.subplots(figsize=(10, 6))  # Increased width for better time step visibility
    
    # Define distinct colors and line styles for each agent
    agent_styles = {
        "Baseline": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "TabularQLearning": {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
        "SARSA": {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
        "ExpectedSARSA": {"color": "#d62728", "linestyle": ":", "marker": "D"},
        "NStepSARSA": {"color": "#9467bd", "linestyle": "-", "marker": "v"},
        "DQN": {"color": "#8c564b", "linestyle": "--", "marker": "*"}
    }

    has_data_for_metric = False
    for agent_name, history in all_metrics.items():
        if not history:
            continue

        # Special handling for epsilon and dqn_loss
        if metric_key == "epsilon" and history[0].get("epsilon", -1.0) == -1.0:
            continue
        if metric_key == "dqn_loss" and agent_name != "DQN":
            continue

        time_steps = np.array([m["time_step"] for m in history])
        metric_values = np.array([m.get(metric_key, np.nan) for m in history])

        # Handle special cases
        if 'sinr' in metric_key.lower() or 'rsrp' in metric_key.lower():
            metric_values[metric_values == -np.inf] = np.nan

        # Filter out NaN values
        valid_indices = ~np.isnan(metric_values)
        if np.any(valid_indices):
            style = agent_styles.get(agent_name, {})
            ax.plot(time_steps[valid_indices], metric_values[valid_indices],
                   label=agent_name,
                   color=style.get("color"),
                   linestyle=style.get("linestyle"),
                   marker=style.get("marker"),
                   markersize=4,
                   linewidth=1.5,
                   markevery=10)  # Show markers every 10 points for better visibility
            has_data_for_metric = True

    # Set title and labels with proper formatting
    ax.set_title(plot_title, fontsize=14, pad=15)
    ax.set_xlabel("Time Step", fontsize=12, labelpad=10)
    ylabel_text = plot_title.split("(")[-1].replace(")", "").strip() if "(" in plot_title else metric_key.replace("_", " ").title()
    ax.set_ylabel(ylabel_text, fontsize=12, labelpad=10)

    # Improve grid and ticks
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Set x-axis ticks to show every 20 steps
    ax.set_xticks(np.arange(0, 201, 20))
    
    # Add legend with proper formatting
    if has_data_for_metric:
        ax.legend(fontsize=10, loc='best', framealpha=0.9, edgecolor='black')
    else:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, fontsize=14,
                color='gray', ha='center', va='center')

    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{metric_key}_comparison.png"
    save_handler.save_figure(fig, plot_filename, experiment_name=exp_folder_name)
    
    plt.close(fig)

def display_comparison_plots(all_metrics, save_handler, exp_folder_name):
    """Displays comparison plots for all agents."""
    if not all_metrics:
        print("No data to display.")
        return

    # Metrics to plot (matching GUI's metrics)
    metrics_to_plot = [
        ("avg_ue_throughput_mbps", "Average UE Throughput (Mbps)"),
        ("percentage_satisfied_ues", "Percentage of Satisfied UEs (%)"),
        ("avg_bs_load_factor", "Average BS Load Factor"),
        ("reward", "Step Reward"),
        ("handovers_this_step", "Handovers per Step"),
        ("total_handovers_cumulative", "Cumulative Handovers"),
        ("sinr_avg_db", "Average SINR (dB)"),
        ("rbs_avg_per_ue", "Average RBs per UE"),
        ("epsilon", "Epsilon Decay"),
        ("dqn_loss", "DQN Loss"),
    ]

    # Create a separate plot for each metric
    for metric_key, plot_title in metrics_to_plot:
        create_research_plot(metric_key, plot_title, all_metrics, save_handler, exp_folder_name)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run RL agent comparison simulations')
    parser.add_argument('--placement', choices=['uniform', 'PPP'], default='uniform',
                      help='Placement method for BSs and UEs (default: uniform)')
    parser.add_argument('--lambda-bs', type=float, default=0.5,
                      help='Lambda parameter for BS PPP placement (default: 0.5)')
    parser.add_argument('--lambda-ue', type=float, default=2.0,
                      help='Lambda parameter for UE PPP placement (default: 2.0)')
    args = parser.parse_args()

    # Setup logging and saving
    log_file = setup_logging()
    save_handler = SaveHandler(output_dir="simulation_results")
    exp_folder_name = f"comparison_run_{args.placement}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Define agents to test
    agents_to_test = [
        "Baseline",
        "TabularQLearning",
        "SARSA",
        "ExpectedSARSA",
        "NStepSARSA",
    ]
    if TF_AVAILABLE_GLOBAL:
        agents_to_test.append("DQN")
    else:
        with open(log_file, 'a') as f:
            with redirect_stdout(f):
                print("Warning: TensorFlow not available. DQN agent will be skipped.")

    # Create base parameters
    params = SimParams()
    
    # Set placement method and parameters
    params.placement_method = args.placement
    if args.placement == "PPP":
        params.lambda_bs = args.lambda_bs
        params.lambda_ue = args.lambda_ue
    
    # Set parameters with reduced steps
    params.total_sim_steps = 200  # Reduced to 200 steps
    params.rl_epsilon_decay_steps = 160  # Adjusted for 200 steps
    params.rl_batch_size = 32
    params.rl_learning_rate = 0.001
    params.rl_target_update_freq = 10  # More frequent updates for shorter training
    params.rl_replay_buffer_size = 2000  # Reduced buffer size
    params.rl_num_hidden_layers = 2
    params.rl_hidden_units = 64
    params.rl_dropout_rate = 0.1
    params.rl_use_soft_updates = True
    params.rl_tau = 0.01

    # Network configuration
    params.num_ues = 10
    params.num_bss = 3
    params.num_total_rbs = 20
    params.target_ue_throughput_mbps = 1.0

    # Run simulations for each agent with progress bar
    all_metrics = {}
    for agent in tqdm(agents_to_test, desc="Running Simulations", colour="green"):
        # Use current timestamp as seed for reproducibility
        seed = int(time.time() * 1000) % 1000000
        metrics = run_single_simulation(agent, params, seed, log_file)
        if metrics:
            all_metrics[agent] = metrics
            # Save metrics to CSV
            save_handler.save_metrics(agent, metrics, params.to_dict(), experiment_name=exp_folder_name)

    # Display comparison plots
    if all_metrics:
        display_comparison_plots(all_metrics, save_handler, exp_folder_name)
        print(f"\nResults saved to: {save_handler.output_dir}/{exp_folder_name}")
    else:
        print("No successful simulations to compare.")

if __name__ == "__main__":
    main() 