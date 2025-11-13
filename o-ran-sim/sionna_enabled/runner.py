"""Runner that demonstrates Sionna integration.

This runner inserts the project root into sys.path so running the script
directly will be able to import the sibling packages like `sim_core` and
`rl_agents`.
"""
import sys
import os
import time

# Make repo root importable when running this file directly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim_core.params import SimParams
from sim_core.simulation import Simulation
from sionna_enabled.sionna_wrapper import sionna_available, init_sionna_channel
import argparse
from sionna_enabled.integration import enable_sionna_fading


def run_smoke_with_sionna(steps=10, use_sionna=False):
    params = SimParams()
    params.total_sim_steps = steps
    params.placement_method = "uniform"

    # Standard logger prints to stdout
    def logger(msg):
        print(msg)

    sim = Simulation(params, logger)

    if use_sionna:
        # Attempt to enable deeper Sionna-based fading integration
        print("Attempting to enable Sionna fading integration...")
        patched = enable_sionna_fading(sim, {"tx": 1, "rx": 1})
        if not patched:
            print("Sionna integration not enabled (Sionna may be missing). Running original simulation.")

    # Run the simulation steps
    while sim.run_step():
        time.sleep(0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sionna", action="store_true", help="Enable Sionna fading integration if available")
    args = parser.parse_args()
    run_smoke_with_sionna(steps=args.steps, use_sionna=args.sionna)
