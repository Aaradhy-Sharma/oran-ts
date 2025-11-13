#!/usr/bin/env python3
"""Run experiments like run_all_experiments.py but support a --sionna flag.

This runner lives in the sandbox and does not modify original project files.
When --sionna is passed it will attempt to enable Sionna fading integration
before each simulation run.
"""
import os
import sys
import time
import argparse
from datetime import datetime

# Ensure repo root is importable when running this script directly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from final_runner import setup_logging, display_comparison_plots
from sionna_enabled.sionna_runner import run_single_simulation_sionna
from sim_core.params import SimParams
from sim_core.constants import TF_AVAILABLE as TF_AVAILABLE_GLOBAL
from utils.saver import SaveHandler

from sionna_enabled.integration import enable_sionna_fading
from sionna_enabled.experiment_spec import SPEC as EXP_SPEC


def run_for_placement(placement: str, out_root: str, total_steps: int, use_sionna: bool, repetition_idx: int = 0):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_folder_name = f"final_comparison_{placement}_{timestamp}"
    results_dir = os.path.join(out_root, exp_folder_name)
    metrics_dir = os.path.join(results_dir, "metrics")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Set up logger file (final_runner will append to this file)
    log_file = setup_logging()

    save_handler = SaveHandler(output_dir=results_dir)

    agents_to_test = [
        "Baseline",
        "TabularQLearning",
        "SARSA",
        "ExpectedSARSA",
        "NStepSARSA",
    ]
    if TF_AVAILABLE_GLOBAL:
        agents_to_test.extend(["DQN", "NO_DECAY_DQN"])

    base_params = SimParams()
    base_params.placement_method = placement
    base_params.total_sim_steps = total_steps

    base_params.rl_batch_size = 32
    base_params.rl_learning_rate = 0.001
    base_params.rl_target_update_freq = 5
    base_params.rl_replay_buffer_size = 2000

    all_metrics = {}

    for agent in agents_to_test:
        # copy params
        agent_params = SimParams()
        for k, v in base_params.to_dict().items():
            setattr(agent_params, k, v)

        if agent == "NO_DECAY_DQN":
            agent_params.rl_epsilon_decay_steps = 0
            agent_params.rl_epsilon_start = 0.1
            agent_params.rl_epsilon_end = 0.1
        else:
            agent_params.rl_epsilon_decay_steps = max(1, int(base_params.total_sim_steps * 0.8))

        # seed per repetition for reproducibility across agents
        seed = int(time.time() * 1000) % 1000000

        # Run via sandbox runner which can enable Sionna per-run
        metrics, returned_params = run_single_simulation_sionna(agent, agent_params, seed, log_file, use_sionna)

        if metrics:
            all_metrics[agent] = metrics
            sim_params_to_save = returned_params if returned_params is not None else agent_params.to_dict()
            save_handler.save_metrics(agent, metrics, sim_params_to_save, experiment_name=None)

    if all_metrics:
        save_handler.save_to_csv(all_metrics, filename="aggregated_metrics.csv", experiment_name=None)
        display_comparison_plots(all_metrics, save_handler, exp_folder_name)
        print(f"Results saved to: {results_dir}")
    else:
        print("No successful simulations to compare for placement:", placement)


def main():
    parser = argparse.ArgumentParser(description="Run experiments for multiple placements and agents (sionna-enabled)")
    parser.add_argument('--placements', nargs='+', default=None, help='List of placements to run')
    parser.add_argument('--out', default='res', help='Root output directory')
    parser.add_argument('--steps', type=int, default=None, help='Total sim steps per run')
    parser.add_argument('--sionna', action='store_true', help='Attempt to enable Sionna integration')
    parser.add_argument('--repetitions', type=int, default=None, help='Override repetitions per placement')

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Use experiment spec defaults unless CLI overrides are provided
    placements_to_run = args.placements if args.placements else list(EXP_SPEC['placements'].keys())

    for placement in placements_to_run:
        spec = EXP_SPEC['placements'].get(placement, {})
        steps = args.steps if args.steps is not None else spec.get('steps', 200)
        repetitions = args.repetitions if args.repetitions is not None else spec.get('repetitions', 1)

        print(f"\n=== Running placement: {placement} (sionna={args.sionna}) repetitions={repetitions} steps={steps} ===")
        for rep in range(repetitions):
            print(f"--- repetition {rep+1}/{repetitions} ---")
            run_for_placement(placement, args.out, steps, args.sionna, repetition_idx=rep)


if __name__ == '__main__':
    main()
