"""Experiment specification used by the sionna-enabled batch runner.

This file encodes the experiment parameterization (placements, step counts,
repetitions, and any per-placement parameter overrides) used to reproduce the
experiments described in the paper (work.pdf).

ASSUMPTIONS:
- The original paper uses per-run step counts of 200 for each experiment. If
  the paper requires different values, edit the `placements` entries below.
- We run a small number of repetitions (default 5) for statistical aggregation.
  You can increase this for larger experiments but expect longer runtimes.

If you want exact values different from these defaults, update this file.
"""

SPEC = {
    # Placements and their parameters. Use 'steps' for number of time steps
    # per simulation, and 'repetitions' for how many independent seeds to run.
    "placements": {
        "uniform": {
            "steps": 200,
            "repetitions": 5,
            "num_bss": 3,
            "num_ues": 10
        },
        "PPP": {
            "steps": 200,
            "repetitions": 5,
            # For PPP we let the simulator sample the number of BS/UEs via lambda
            "num_bss": None,
            "num_ues": None
        }
    },

    # Agents to include in experiments. DQN agents will be skipped if TF/Sionna
    # is not available in the environment.
    "agents": [
        "Baseline",
        "TabularQLearning",
        "SARSA",
        "ExpectedSARSA",
        "NStepSARSA",
        "DQN",
        "NO_DECAY_DQN",
    ],

    # Random seed strategy: 'time' uses a time-based seed per repetition, or you
    # can supply an explicit list of seeds under 'fixed_seeds'. If fixed_seeds
    # is provided it should be a list of integers with length >= repetitions.
    "seed_mode": "time",
    "fixed_seeds": None,
}
