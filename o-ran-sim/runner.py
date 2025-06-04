# runner.py

import random
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time # For pausing if needed
import collections # For defaultdict in aggregation
import traceback # For detailed error logging
import pandas as pd # Make sure pandas is imported here if you don't solely rely on saver.py
import math # Needed for dynamic plot grid size and possibly scenario generation

# Import core simulation components
from sim_core.params import SimParams
from sim_core.simulation import Simulation
from sim_core.constants import TF_AVAILABLE as TF_AVAILABLE_GLOBAL # Check if TF is available for DQN

# Import utilities
from utils.logger import LogHandler
from utils.saver import SaveHandler

# --- Setup for Automated Runs ---
log_handler_runner = LogHandler()
save_handler_runner = SaveHandler(output_dir="automated_sim_results") # Dedicated directory for automated runs
# Set up file logging for the runner script
log_file_path = os.path.join(save_handler_runner.output_dir, f"runner_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
log_handler_runner.set_file_logger(log_file_path)
log_handler_runner.disable_gui_logging() # Ensure GUI logging is off for runner script

def run_single_simulation_experiment(
    agent_type: str,
    base_params_template: SimParams, # Renamed to emphasize it's a template
    experiment_overrides: dict,
    run_seed: int,
    exp_name: str, # Overall experiment name for logging context
):
    """
    Runs a single simulation with a specific agent type and parameter overrides.
    Returns a tuple: (metrics_history, final_sim_params_dict).
    """
    log_handler_runner.log(f"\n--- Running: {agent_type} for {exp_name} (Seed: {run_seed}) ---", level='info')

    # Create a new SimParams object for this specific run
    current_params = SimParams()
    # Copy attributes from the template
    for attr, value in base_params_template.to_dict().items():
        setattr(current_params, attr, value)

    # Apply scenario-specific overrides
    for param_name, param_value in experiment_overrides.items():
        if hasattr(current_params, param_name):
            setattr(current_params, param_name, param_value)
        else:
            log_handler_runner.log(f"Warning: Attempted to override non-existent parameter '{param_name}'.", level='warning')
    
    # Ensure derived parameters are calculated based on potentially new values
    current_params.rb_bandwidth_hz = current_params.rb_bandwidth_mhz * 1e6
    current_params.ho_time_to_trigger_steps = (
        int(current_params.ho_time_to_trigger_s / current_params.time_step_duration)
        if current_params.time_step_duration > 0
        else 0
    )
    # Set the RL agent type specifically for this simulation instance
    current_params.rl_agent_type = agent_type

    log_handler_runner.log(f"SimParams for this run: {current_params.to_dict()}", level='debug')
    
    # Set random seeds for this specific run
    random.seed(run_seed)
    np.random.seed(run_seed)
    if TF_AVAILABLE_GLOBAL:
        import tensorflow as tf
        tf.random.set_seed(run_seed)

    simulation = None
    try:
        simulation = Simulation(current_params, log_handler_runner.log)
        log_handler_runner.log(f"Simulation setup for {agent_type} complete. Starting steps...", level='info')

        for step in range(current_params.total_sim_steps):
            if not simulation.run_step():
                log_handler_runner.log(f"Simulation for {agent_type} ended early at step {step}.", level='info')
                break
            # log_handler_runner.log(f"Step {step} complete.", level='debug') # Can be very verbose for long runs

        log_handler_runner.log(f"Simulation for {agent_type} completed {simulation.current_time_step} steps.", level='info')
        # Return metrics and the final parameters dictionary
        return simulation.metrics_history, current_params.to_dict()

    except ImportError as e:
        log_handler_runner.log(f"ERROR: Failed to run {agent_type} due to missing library: {e}", level='error')
        return None, None
    except Exception as e:
        log_handler_runner.log(f"ERROR: Simulation for {agent_type} failed: {e}", level='error')
        # import traceback # Already imported at the top
        log_handler_runner.log(traceback.format_exc(), level='error')
        return None, None
    finally:
        # Crucial to close figures opened by matplotlib during visualization/rendering in sim loop
        plt.close('all')


def generate_comparison_plots(all_experiments_metrics, save_plots=True, exp_folder_name="comparison_plots"):
    """
    Generates comparison plots for multiple agents across different experimental conditions.
    all_experiments_metrics: {'exp_name': {'agent_type': [metrics_history]}}
    """
    log_handler_runner.log(f"\n--- Generating Comparison Plots ---", level='info')

    # Metrics to plot (can be customized)
    metrics_to_plot = [
        ("avg_ue_throughput_mbps", "Avg UE Throughput (Mbps)"),
        ("percentage_satisfied_ues", "% Satisfied UEs"),
        ("avg_bs_load_factor", "Avg BS Load Factor"),
        ("reward", "Step Reward"),
        ("handovers_this_step", "Handovers per Step"),
        ("total_handovers_cumulative", "Cumulative Handovers"),
        ("sinr_avg_db", "Average SINR (dB)"),
        ("rbs_avg_per_ue", "Average RBs per UE"),
        ("epsilon", "Epsilon Decay"), # Only relevant for RL agents
    ]

    for exp_name, agents_data in all_experiments_metrics.items():
        if not agents_data:
            log_handler_runner.log(f"No data for experiment {exp_name}. Skipping plots.", level='warning')
            continue

        # Determine how many plots we'll actually draw
        num_actual_plots = 0
        for metric_key, _ in metrics_to_plot:
            # Check if at least one agent has non-NaN data for this metric
            for agent_name, history in agents_data.items():
                if history and metric_key in history[0]:
                    values = [m.get(metric_key) for m in history if m.get(metric_key) is not None]
                    if metric_key == "epsilon" and all(v == -1.0 for v in values if isinstance(v, (int, float))):
                        continue # Skip epsilon plot if all values are -1.0
                    if any(not (isinstance(v, (float, int)) and (np.isnan(v) or np.isinf(v))) for v in values):
                        num_actual_plots += 1
                        break # Found at least one agent with valid data for this metric
        
        if num_actual_plots == 0:
            log_handler_runner.log(f"No plottable data found for experiment {exp_name}.", level='warning')
            continue

        # Use fixed 3x3 for consistency or adjust based on actual plots
        n_rows = 3
        n_cols = 3

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 15))
        # Flatten axs for easy iteration if it's a 2D array, otherwise just use it
        if isinstance(axs, np.ndarray):
            axs = axs.flatten()
        else: # Handle case of single subplot (axs is not an array)
            axs = [axs] 
            
        fig.suptitle(f"Experiment: {exp_name} - Agent Performance Over Time", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96], pad=4.0)

        current_plot_idx = 0
        for i, (metric_key, plot_title) in enumerate(metrics_to_plot):
            if current_plot_idx >= len(axs): # Ensure we don't go out of bounds of created axes
                break
            ax = axs[current_plot_idx]

            has_data_for_metric = False
            for agent_name, history in agents_data.items():
                if not history:
                    continue
                
                # Special handling for epsilon: only plot if it's an RL agent with actual epsilon values
                if metric_key == "epsilon":
                    # epsilon is -1.0 for Baseline, and also for some RL agents if not implemented or setup
                    if history[0].get("epsilon", -1.0) == -1.0: # Check the first step for epsilon default
                        continue
                    
                time_steps = np.array([m["time_step"] for m in history])
                # Use .get() with default value to avoid KeyError if metric is missing in some steps
                metric_values = np.array([m.get(metric_key, np.nan) for m in history])

                # Replace -inf with NaN for plotting SINR or RSRP if it skews the axis
                if 'sinr' in metric_key.lower() or 'rsrp' in metric_key.lower():
                    metric_values[metric_values == -np.inf] = np.nan
                
                # Filter out NaN values for plotting (both time_steps and values)
                valid_indices = ~np.isnan(metric_values)
                if np.any(valid_indices): # Only plot if there's valid data after filtering
                    ax.plot(time_steps[valid_indices], metric_values[valid_indices], marker=".", linestyle="-", markersize=3, label=agent_name)
                    has_data_for_metric = True
                
            ax.set_title(plot_title)
            ax.set_xlabel("Time Step")
            ylabel_text = plot_title.split("(")[-1].replace(")", "").strip() if "(" in plot_title else metric_key.replace("_", " ").title()
            ax.set_ylabel(ylabel_text)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            if has_data_for_metric:
                ax.legend(fontsize="small", loc='best')
            else:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, fontsize=14, color='gray', ha='center', va='center')
            
            current_plot_idx += 1 # Move to the next subplot

        # Hide any unused subplots
        for j in range(current_plot_idx, len(axs)):
            fig.delaxes(axs[j])


        if save_plots:
            # Generate a unique filename for the plot
            plot_filename = f"{exp_folder_name}_{exp_name}_performance_comparison.png"
            save_handler_runner.save_figure(fig, plot_filename, experiment_name=exp_folder_name)
        
        plt.show(block=False) # Use block=False to allow runner to continue immediately
        plt.pause(0.1) # Small pause to ensure plot rendering
        plt.close(fig) # Close the figure immediately after showing/saving to free up memory

# --- Experiment Definitions ---

# Define the agents to test
AGENTS_TO_TEST = [
    "Baseline",
    "TabularQLearning",
    "SARSA",
    "ExpectedSARSA",
    "NStepSARSA",
]
if TF_AVAILABLE_GLOBAL:
    AGENTS_TO_TEST.append("DQN")
else:
    log_handler_runner.log("DQN agent skipped as TensorFlow is not available.", level='warning')


# Define the base parameters for the simulation
base_sim_params = SimParams()
base_sim_params.total_sim_steps = 300 # Increased steps for more learning
# Epsilon decay steps should be absolute steps, not a fraction of total_sim_steps,
# as total_sim_steps can vary by experiment. Set it as a fixed large number or proportional
# to the *default* total_sim_steps. Or, calculate it for each run based on *its* total_sim_steps.
# For now, we set it here in base_sim_params for initial value, but it's *overridden* within run_single_simulation_experiment
base_sim_params.rl_epsilon_decay_steps = 240 # Decay over 80% of 300 steps


# Define different experiment scenarios (parameter variations)
# Each item in the list is one experiment scenario.
# The dict within each item specifies parameter overrides for that scenario.
EXPERIMENT_SCENARIOS = {
    "Default_Config": {
        "num_ues": 10,
        "num_bss": 3,
        "placement_method": "Uniform Random",
        "ue_speed_mps": 5.0,
        "num_total_rbs": 20,
    },
    "High_Density_Dynamic": {
        "num_ues": 30, # High UE count
        "num_bss": 5,  # More BSs
        "lambda_ue": 30 / (1000*1000*1e-6), # For PPP placement
        "lambda_bs": 5 / (1000*1000*1e-6),  # For PPP placement
        "placement_method": "PPP",
        "ue_speed_mps": 8.0, # Increased mobility
        "num_total_rbs": 50, # More RBs to handle density
        "shadowing_std_dev_db": 6.0, # More channel variability
        "rl_learning_rate": 0.005 # Potentially faster learning rate for DRL
    },
    "BS_Hotspot_Uneven_Load": {
        "num_ues": 25,
        "num_bss": 4,
        "placement_method": "Uniform Random",
        # For this scenario, BS placement would ideally be clustered, but with random
        # placement we simulate high demand in certain areas by increasing UE count.
        "ue_speed_mps": 4.0, # Slower mobility to expose load issues
        "num_total_rbs": 30,
        "target_ue_throughput_mbps": 1.5, # Higher demands
        "ho_hysteresis_db": 4.0, # Less aggressive baseline HO
    },
    "Sparse_Network_Challenging_Coverage": {
        "num_ues": 15,
        "num_bss": 2, # Fewer BSs for coverage challenge
        "placement_method": "Uniform Random",
        "ue_speed_mps": 7.0, # Faster mobility, harder to maintain connection
        "num_total_rbs": 20,
        "path_loss_exponent": 4.0, # More severe path loss
        "shadowing_std_dev_db": 5.0, # More variability in coverage
    },
    "High_Traffic_Low_Resources": {
        "num_ues": 20,
        "num_bss": 3,
        "placement_method": "Uniform Random",
        "ue_speed_mps": 5.0,
        "num_total_rbs": 15, # Limited resources
        "target_ue_throughput_mbps": 1.2, # Higher demand per UE
        "rb_bandwidth_mhz": 0.4, # Smaller RB bandwidth
    },
    "Extreme_Mobility": {
        "num_ues": 10,
        "num_bss": 3,
        "placement_method": "Uniform Random",
        "ue_speed_mps": 15.0, # Very high mobility
        "time_step_duration": 0.1, # Shorter time steps to capture rapid changes
        "total_sim_steps": 600, # More steps needed for high mobility
        "rl_epsilon_decay_steps": 480, # Adjust decay for longer run
    }
}

# Number of random seeds to run for each scenario-agent combination (for robustness)
NUM_SEEDS_PER_RUN = 3 # Increased for better statistical significance

# --- Main Runner Logic ---
if __name__ == "__main__":
    # This structure will store results like:
    # {'scenario_name': {'agent_type': [{'time_step':0, ...}, {'time_step':1, ...}]}}
    # Using collections.defaultdict for easier appending
    all_experiment_results = collections.defaultdict(lambda: collections.defaultdict(list))
    exp_folder_name = f"Experiment_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # Unique folder for this entire run

    log_handler_runner.log(f"Starting Automated Experiments. Output will be saved to: {save_handler_runner.output_dir}/{exp_folder_name}", level='info')
    log_handler_runner.log(f"Testing Agents: {AGENTS_TO_TEST}", level='info')
    log_handler_runner.log(f"Experiment Scenarios: {list(EXPERIMENT_SCENARIOS.keys())}", level='info')
    log_handler_runner.log(f"Runs per Scenario-Agent: {NUM_SEEDS_PER_RUN}", level='info')

    # Iterate through each experiment scenario
    for scenario_name, overrides in EXPERIMENT_SCENARIOS.items():
        log_handler_runner.log(f"\n===== Running Scenario: {scenario_name} =====", level='info')
        
        # This will store the aggregated metrics for each agent within this scenario
        scenario_aggregated_metrics = collections.defaultdict(list) # {'agent_type': [aggregated_metric_dict_per_step]}

        # Iterate through each agent type
        for agent in AGENTS_TO_TEST:
            agent_metrics_across_seeds = []
            
            # Run multiple times with different seeds for statistical robustness
            for i in range(NUM_SEEDS_PER_RUN):
                # Using current timestamp as part of seed for more entropy in each unique run
                # Ensure it's an integer for random.seed
                seed = int(time.time() * 1000 + i) % 1000000 
                
                log_handler_runner.log(f"Starting run {i+1}/{NUM_SEEDS_PER_RUN} for {agent} in {scenario_name} (Seed: {seed})...", level='info')
                
                metrics, final_params_dict = run_single_simulation_experiment(
                    agent, base_sim_params, overrides, seed, scenario_name
                )
                
                if metrics:
                    agent_metrics_across_seeds.append(metrics)
                    # Save raw metrics for each individual run into its specific experiment subfolder
                    run_unique_id = f"{scenario_name}_run{i+1}_seed{seed}"
                    save_handler_runner.save_metrics(
                        agent, metrics, final_params_dict,
                        experiment_name=os.path.join(exp_folder_name, "raw_metrics_per_run", run_unique_id)
                    )
                else:
                    log_handler_runner.log(f"Run {i+1} for {agent} in {scenario_name} failed. Skipping metrics for this run.", level='warning')
            
            # --- Aggregate metrics across seeds for this agent and scenario ---
            if agent_metrics_across_seeds:
                # Use defaultdict of lists to collect all values for each metric at each step
                # Temporarily store values by step number for averaging
                temp_aggregated_by_step_and_metric = collections.defaultdict(lambda: collections.defaultdict(list))

                # Aggregate all runs for this agent within this scenario
                for history in agent_metrics_across_seeds:
                    for step_data in history:
                        step_num = step_data.get("time_step", -1)
                        if step_num == -1: continue # Skip if time_step is somehow missing

                        for key, value in step_data.items():
                            # Only aggregate numerical values that are not NaN/Inf for averaging
                            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                                temp_aggregated_by_step_and_metric[step_num][key].append(value)
                            # Handle non-numeric or specific keys (e.g., ue_details, which shouldn't be averaged)
                            elif key not in temp_aggregated_by_step_and_metric[step_num]:
                                temp_aggregated_by_step_and_metric[step_num][key] = value # Just take the first if not for averaging

                # Final averaging for each step
                avg_metrics_history_for_agent = []
                # Ensure we iterate through steps in order based on available data
                sorted_step_nums = sorted(temp_aggregated_by_step_and_metric.keys())

                for step_num in sorted_step_nums:
                    avg_step = {"time_step": step_num}
                    
                    # Get a reference list of keys from the first available history for this agent type
                    # This helps ensure all metrics are considered, even if some have no data to average
                    reference_keys = agent_metrics_across_seeds[0][0].keys() if agent_metrics_across_seeds and agent_metrics_across_seeds[0] else []

                    for key in reference_keys:
                        values_to_avg = temp_aggregated_by_step_and_metric[step_num].get(key)
                        
                        if isinstance(values_to_avg, list) and values_to_avg: # If it's a list of numbers to average
                            avg_step[key] = np.mean(values_to_avg)
                        elif values_to_avg is not None: # If it's a single non-list value (e.g., 'ue_details')
                            avg_step[key] = values_to_avg
                        else: # If no data for this key at this step or default value
                            if key == "epsilon": avg_step[key] = -1.0 # Default for epsilon
                            elif key == "ue_details": avg_step[key] = [] # Default for ue_details
                            # Attempt to use a default for numeric types if possible, otherwise NaN
                            elif isinstance(getattr(base_sim_params, key, None), (int, float)):
                                avg_step[key] = np.nan
                            else:
                                avg_step[key] = None # Or some other appropriate default
                    avg_metrics_history_for_agent.append(avg_step)
                
                scenario_aggregated_metrics[agent] = avg_metrics_history_for_agent
            else:
                log_handler_runner.log(f"No successful runs for agent {agent} in scenario {scenario_name}. Skipping aggregation.", level='warning')
        
        # Store the aggregated results for this scenario
        all_experiment_results[scenario_name] = scenario_aggregated_metrics

        # NEW: Save aggregated metrics for this scenario to CSV
        if scenario_aggregated_metrics:
            csv_filename = f"{scenario_name}_aggregated_metrics.csv"
            save_handler_runner.save_to_csv(
                scenario_aggregated_metrics,
                csv_filename,
                experiment_name=exp_folder_name # Save inside the main experiment run folder
            )
        else:
            log_handler_runner.log(f"No aggregated data for scenario {scenario_name}. Skipping CSV save.", level='warning')


    # Generate and optionally save all comparison plots
    log_handler_runner.log("\n--- All Experiments Finished ---", level='info')
    
    # Prompt user for plot saving and display
    save_plots_choice = input("\nDo you want to save the comparison plots to disk? (y/n): ").lower()
    save_plots = save_plots_choice == 'y'

    if save_plots:
        log_handler_runner.log(f"Saving plots to {save_handler_runner.output_dir}/{exp_folder_name}", level='info')
    else:
        log_handler_runner.log("Plots will not be saved to disk.", level='info')
    
    # Generate and display plots for each scenario using the aggregated data
    for scenario_name, agents_data in all_experiment_results.items():
        if agents_data:
             generate_comparison_plots(
                {scenario_name: agents_data}, # Pass a dictionary for just this scenario
                save_plots=save_plots,
                exp_folder_name=exp_folder_name # Pass overall folder name for saving path
            )
        else:
            log_handler_runner.log(f"Skipping plot generation for scenario {scenario_name} due to no data.", level='warning')

    log_handler_runner.log(f"Automated experiments completed. Full log saved to: {log_file_path}", level='info')
    print(f"\nAutomated experiments completed. Results are in the '{save_handler_runner.output_dir}' directory.")
    print(f"Full log file: {log_file_path}")