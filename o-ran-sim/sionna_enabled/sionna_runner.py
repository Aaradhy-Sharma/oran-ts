"""Sandbox runner that constructs Simulation objects and can enable Sionna integration per run.

This mirrors `final_runner.run_single_simulation` but stays in the sandbox and
applies `enable_sionna_fading` when requested. It does not modify original
project files.
"""
import random
import time
import numpy as np
from contextlib import redirect_stdout
from datetime import datetime
from sim_core.simulation import Simulation
from sionna_enabled.integration import enable_sionna_fading


def run_single_simulation_sionna(agent_type, params, run_seed, log_file, use_sionna=False):
    """Run a single simulation instance and optionally enable Sionna fading.

    Returns metrics_history or None on error.
    """
    try:
        with open(log_file, 'a') as f:
            with redirect_stdout(f):
                print(f"\n--- Running (sionna): {agent_type} (Seed: {run_seed}) ---")
                print(f"Placement Method: {params.placement_method}")

                # Set seeds
                random.seed(run_seed)
                np.random.seed(run_seed)

                params.rl_agent_type = agent_type

                # Construct simulation
                sim = Simulation(params, print)
                print(f"Simulation setup for {agent_type} complete. (sionna={use_sionna})")

                # If requested, enable Sionna fading on this Simulation
                if use_sionna:
                    enable_sionna_fading(sim, {"tx": 1, "rx": 1})

                for step in range(params.total_sim_steps):
                    if not sim.run_step():
                        print(f"Simulation ended early at step {step}.")
                        break

                print(f"Simulation completed {sim.current_time_step} steps.")
                # Embed use_sionna flag into params dict for SaveHandler
                params_dict = params.to_dict() if hasattr(params, 'to_dict') else {}
                params_dict['use_sionna'] = bool(use_sionna)
                return sim.metrics_history, params_dict
    except Exception as e:
        with open(log_file, 'a') as f:
            with redirect_stdout(f):
                print(f"ERROR: Simulation failed: {e}")
    return None, None
