"""Simple smoke test for the Baseline agent and Simulation."""
from sim_core.params import SimParams
from sim_core.simulation import Simulation
from rl_agents import BaselineAgent


def simple_print_logger(msg):
    print(msg)


def test_baseline_smoke():
    params = SimParams()
    params.total_sim_steps = 3
    params.num_ues = 6
    params.num_bss = 2
    params.rl_agent_type = "Baseline"

    sim = Simulation(params, simple_print_logger)
    steps = 0
    while sim.run_step():
        steps += 1
    print("Baseline smoke: steps executed:", sim.current_time_step)
    assert sim.current_time_step > 0


if __name__ == "__main__":
    test_baseline_smoke()
