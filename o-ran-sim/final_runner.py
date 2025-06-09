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
import pandas as pd

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
    log_dir = "FINAL/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"final_simulation_{timestamp}.log")
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
                simulation = Simulation(params, print)
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

def create_research_plot(metric_key, plot_title, all_metrics, save_handler, plots_dir):
    """Creates a single research-quality plot for a specific metric."""
    plt.style.use('seaborn-v0_8-paper')
    
    # Create figure with specific size for research papers
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Define distinct colors and line styles for each agent
    agent_styles = {
        "Baseline": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "TabularQLearning": {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
        "SARSA": {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
        "ExpectedSARSA": {"color": "#d62728", "linestyle": ":", "marker": "D"},
        "NStepSARSA": {"color": "#9467bd", "linestyle": "-", "marker": "v"},
        "DQN": {"color": "#8c564b", "linestyle": "--", "marker": "*"},
        "NO_DECAY_DQN": {"color": "#e377c2", "linestyle": "-", "marker": "p"}
    }

    has_data_for_metric = False
    all_time_steps = set()
    agent_data = {}

    # First pass: collect all time steps and prepare data
    for agent_name, history in all_metrics.items():
        if not history:
            continue

        if metric_key == "epsilon" and history[0].get("epsilon", -1.0) == -1.0:
            continue
        if metric_key == "dqn_loss" and agent_name not in ["DQN", "NO_DECAY_DQN"]:
            continue

        time_steps = np.array([m["time_step"] for m in history])
        metric_values = np.array([m.get(metric_key, np.nan) for m in history])

        # Special handling for percentage_satisfied_ues
        if metric_key == "percentage_satisfied_ues":
            num_satisfied = np.array([m.get("num_satisfied_ues", 0) for m in history])
            num_connected = np.array([m.get("num_connected_ues", 1) for m in history])
            metric_values = (num_satisfied / num_connected) * 100
            metric_values = np.clip(metric_values, 0, 100)

        if 'sinr' in metric_key.lower() or 'rsrp' in metric_key.lower():
            metric_values = np.where(metric_values == -np.inf, np.nan, metric_values)

        valid_indices = ~np.isnan(metric_values)
        if np.any(valid_indices):
            all_time_steps.update(time_steps[valid_indices])
            agent_data[agent_name] = {
                'time_steps': time_steps[valid_indices],
                'values': metric_values[valid_indices]
            }
            has_data_for_metric = True

    # Convert to sorted list for consistent plotting
    all_time_steps = sorted(list(all_time_steps))

    # Plot line graph (top subplot)
    for agent_name, data in agent_data.items():
        style = agent_styles.get(agent_name, {})
        ax1.plot(data['time_steps'], data['values'],
               label=agent_name,
               color=style.get("color"),
               linestyle=style.get("linestyle"),
               marker=style.get("marker"),
               markersize=4,
               linewidth=1.5,
               markevery=10)

    # Set title and labels for line plot
    ax1.set_title(f"{plot_title} Over Time", fontsize=14, pad=15)
    ax1.set_xlabel("Time Step", fontsize=12, labelpad=10)
    ylabel_text = plot_title.split("(")[-1].replace(")", "").strip() if "(" in plot_title else metric_key.replace("_", " ").title()
    ax1.set_ylabel(ylabel_text, fontsize=12, labelpad=10)

    # Set y-axis limits for percentage_satisfied_ues
    if metric_key == "percentage_satisfied_ues":
        ax1.set_ylim(0, 100)

    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_xticks(np.arange(0, 201, 20))

    # Create bar plot (bottom subplot)
    if has_data_for_metric:
        # Select time steps to plot (every 20th step to avoid overcrowding)
        plot_time_steps = all_time_steps[::20]
        if not plot_time_steps:
            plot_time_steps = all_time_steps

        # Prepare data for grouped bars
        x = np.arange(len(plot_time_steps))
        width = 0.1  # Width of bars
        agents = list(agent_data.keys())
        
        # Plot grouped bars
        for i, agent_name in enumerate(agents):
            data = agent_data[agent_name]
            # Interpolate values for selected time steps
            values = np.interp(plot_time_steps, data['time_steps'], data['values'])
            offset = (i - len(agents)/2 + 0.5) * width
            ax2.bar(x + offset, values, width, label=agent_name, 
                   color=agent_styles[agent_name]["color"])

        ax2.set_title(f"{plot_title} at Selected Time Steps", fontsize=14, pad=15)
        ax2.set_xlabel("Time Step", fontsize=12, labelpad=10)
        ax2.set_ylabel(ylabel_text, fontsize=12, labelpad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(plot_time_steps)
        
        # Set y-axis limits for percentage_satisfied_ues in bar plot
        if metric_key == "percentage_satisfied_ues":
            ax2.set_ylim(0, 100)
        
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        ax2.tick_params(axis='both', which='major', labelsize=10)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    else:
        ax2.text(0.5, 0.5, 'No Data', transform=ax2.transAxes, fontsize=14,
                color='gray', ha='center', va='center')

    if has_data_for_metric:
        ax1.legend(fontsize=10, loc='best', framealpha=0.9, edgecolor='black')
        ax2.legend(fontsize=10, loc='best', framealpha=0.9, edgecolor='black')
    else:
        ax1.text(0.5, 0.5, 'No Data', transform=ax1.transAxes, fontsize=14,
                color='gray', ha='center', va='center')

    plt.tight_layout()
    
    # Save the plot with LaTeX-style naming
    plot_filename = f"fig_{metric_key}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def display_comparison_plots(all_metrics, save_handler, exp_folder_name):
    """Displays comparison plots for all agents."""
    if not all_metrics:
        print("No data to display.")
        return

    metrics_to_plot = [
        ("avg_ue_throughput_mbps", "Average UE Throughput (Mbps)"),
        ("percentage_satisfied_ues", "Percentage of Satisfied UEs (%)"),
        ("avg_bs_load_factor", "Average BS Load Factor"),
        ("reward", "Step Reward"),
        ("handovers_this_step", "Handovers per Step"),
        ("total_handovers_cumulative", "Cumulative Handovers"),
        ("sinr_avg_db", "Average SINR (dB)"),
        ("rbs_avg_per_ue", "Average RBs per UE"),
        ("epsilon", "Epsilon Decay")
    ]

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(save_handler.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for metric_key, plot_title in metrics_to_plot:
        create_research_plot(metric_key, plot_title, all_metrics, save_handler, plots_dir)

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive RL agent comparison simulations')
    parser.add_argument('--placement', choices=['uniform', 'PPP'], default='uniform',
                      help='Placement method for BSs and UEs (default: uniform)')
    parser.add_argument('--lambda-bs', type=float, default=0.5,
                      help='Lambda parameter for BS PPP placement (default: 0.5)')
    parser.add_argument('--lambda-ue', type=float, default=2.0,
                      help='Lambda parameter for UE PPP placement (default: 2.0)')
    args = parser.parse_args()

    # Setup logging and saving
    log_file = setup_logging()
    
    # Create results directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_folder_name = f"final_comparison_{args.placement}_{timestamp}"
    results_dir = os.path.join("FINAL", "results", exp_folder_name)
    metrics_dir = os.path.join(results_dir, "metrics")
    plots_dir = os.path.join(results_dir, "plots")
    
    # Create all necessary directories
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    save_handler = SaveHandler(output_dir=results_dir)
    
    # Define agents to test
    agents_to_test = [
        "Baseline",
        "TabularQLearning",
        "SARSA",
        "ExpectedSARSA",
        "NStepSARSA",
    ]
    if TF_AVAILABLE_GLOBAL:
        agents_to_test.extend(["DQN", "NO_DECAY_DQN"])
    else:
        with open(log_file, 'a') as f:
            with redirect_stdout(f):
                print("Warning: TensorFlow not available. DQN agents will be skipped.")

    # Create base parameters
    params = SimParams()
    
    # Set placement method and parameters
    params.placement_method = args.placement
    if args.placement == "PPP":
        params.lambda_bs = args.lambda_bs
        params.lambda_ue = args.lambda_ue
    
    # Set parameters with reduced steps
    params.total_sim_steps = 200
    params.rl_batch_size = 32
    params.rl_learning_rate = 0.001
    params.rl_target_update_freq = 5
    params.rl_replay_buffer_size = 2000
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
        # Create a copy of params for this agent
        agent_params = SimParams()
        for attr, value in params.to_dict().items():
            setattr(agent_params, attr, value)

        # Special handling for NO_DECAY_DQN
        if agent == "NO_DECAY_DQN":
            agent_params.rl_epsilon_decay_steps = 0
            agent_params.rl_epsilon_start = 0.1
            agent_params.rl_epsilon_end = 0.1
        else:
            agent_params.rl_epsilon_decay_steps = 160

        # Use current timestamp as seed for reproducibility
        seed = int(time.time() * 1000) % 1000000
        metrics = run_single_simulation(agent, agent_params, seed, log_file)
        if metrics:
            all_metrics[agent] = metrics
            # Save metrics to CSV in the metrics directory
            metrics_file = os.path.join(metrics_dir, f"{agent}_metrics.csv")
            pd.DataFrame(metrics).to_csv(metrics_file, index=False)

    # Display comparison plots
    if all_metrics:
        display_comparison_plots(all_metrics, save_handler, exp_folder_name)
        print(f"\nResults saved to: {results_dir}")
    else:
        print("No successful simulations to compare.")

if __name__ == "__main__":
    main() 