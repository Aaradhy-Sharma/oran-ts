#!/usr/bin/env python3
"""Run full experiments for multiple placements and save results under `res/`.

This script imports the helper functions in `final_runner.py` so the
core experiment code remains in one place. It will:
 - Create a timestamped experiment folder under `res/` for each placement
 - Run each agent (skips DQN if TensorFlow not available)
 - Save per-agent JSON metrics via SaveHandler and an aggregated CSV
 - Generate and save comparison plots using the existing plotting

Usage:
    python run_all_experiments.py --placements uniform PPP --out res --steps 200

"""
import os
import time
import argparse
from datetime import datetime

from final_runner import setup_logging, run_single_simulation, display_comparison_plots
from sim_core.params import SimParams
from sim_core.constants import TF_AVAILABLE as TF_AVAILABLE_GLOBAL
from utils.saver import SaveHandler


def run_for_placement(placement: str, out_root: str, total_steps: int):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_folder_name = f"final_comparison_{placement}_{timestamp}"
    results_dir = os.path.join(out_root, exp_folder_name)
    metrics_dir = os.path.join(results_dir, "metrics")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Set up logger file (final_runner will also append to this file)
    log_file = setup_logging()

    save_handler = SaveHandler(output_dir=results_dir)

    # Define agents to test (same as final_runner)
    agents_to_test = [
        "Baseline",
        "TabularQLearning",
        "SARSA",
        "ExpectedSARSA",
        "NStepSARSA",
    ]
    if TF_AVAILABLE_GLOBAL:
        agents_to_test.extend(["DQN", "NO_DECAY_DQN"])

    # Base params
    base_params = SimParams()
    base_params.placement_method = placement
    base_params.total_sim_steps = total_steps

    # Tunable defaults (kept modest so runs finish in reasonable time)
    base_params.rl_batch_size = 32
    base_params.rl_learning_rate = 0.001
    base_params.rl_target_update_freq = 5
    base_params.rl_replay_buffer_size = 2000

    all_metrics = {}

    for agent in agents_to_test:
        # Copy params
        agent_params = SimParams()
        for k, v in base_params.to_dict().items():
            setattr(agent_params, k, v)

        if agent == "NO_DECAY_DQN":
            agent_params.rl_epsilon_decay_steps = 0
            agent_params.rl_epsilon_start = 0.1
            agent_params.rl_epsilon_end = 0.1
        else:
            agent_params.rl_epsilon_decay_steps = max(1, int(base_params.total_sim_steps * 0.8))

        # Use current timestamp as seed for reproducibility per run
        seed = int(time.time() * 1000) % 1000000

        metrics = run_single_simulation(agent, agent_params, seed, log_file)
        if metrics:
            all_metrics[agent] = metrics
            # Save per-agent metrics JSON via SaveHandler
            save_handler.save_metrics(agent, metrics, agent_params.to_dict(), experiment_name=None)

    # Save aggregated CSV
    if all_metrics:
        save_handler.save_to_csv(all_metrics, filename="aggregated_metrics.csv", experiment_name=None)
        display_comparison_plots(all_metrics, save_handler, exp_folder_name)
        print(f"Results saved to: {results_dir}")
    else:
        print("No successful simulations to compare for placement:", placement)


def main():
    parser = argparse.ArgumentParser(description="Run experiments for multiple placements and agents")
    parser.add_argument('--placements', nargs='+', default=['uniform', 'PPP'],
                        help='List of placements to run (e.g., uniform PPP)')
    parser.add_argument('--out', default='res', help='Root output directory to create experiment folders')
    parser.add_argument('--steps', type=int, default=200, help='Total simulation steps per run')

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for placement in args.placements:
        print(f"\n=== Running placement: {placement} ===")
        run_for_placement(placement, args.out, args.steps)


if __name__ == '__main__':
    main()
