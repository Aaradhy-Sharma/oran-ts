# ... existing imports ...

# Define the agents to test
AGENTS_TO_TEST = [
    "Baseline",
    "TabularQLearning",
    "SARSA",
    "ExpectedSARSA",
    "NStepSARSA",
    "DQN"  # Add DQN directly to the list instead of conditionally
]

# Define the base parameters for the simulation
base_sim_params = SimParams()
base_sim_params.total_sim_steps = 500  # Increased from 300
base_sim_params.time_step_duration = 0.1  # Decreased from 0.2
base_sim_params.rl_epsilon_decay_steps = 400  # Adjusted for new total_sim_steps

# Add DQN-specific parameters to base params
base_sim_params.rl_learning_rate = 0.0001
base_sim_params.rl_batch_size = 64
base_sim_params.rl_replay_buffer_size = 10000
base_sim_params.rl_target_update_freq = 10
base_sim_params.rl_gamma = 0.99
base_sim_params.rl_hidden_units = 128
base_sim_params.rl_num_hidden_layers = 3
base_sim_params.rl_dropout_rate = 0.2

# Update experiment scenarios
EXPERIMENT_SCENARIOS = {
    "Default_Config": {
        "num_ues": 10,
        "num_bss": 3,
        "placement_method": "Uniform Random",
        "ue_speed_mps": 5.0,
        "num_total_rbs": 20,
        "rl_learning_rate": 0.0001,
        "rl_batch_size": 64,
        "rl_replay_buffer_size": 10000,
        "rl_target_update_freq": 10,
        "rl_gamma": 0.99,
        "rl_hidden_units": 128,
        "rl_num_hidden_layers": 3,
        "rl_dropout_rate": 0.2,
        "rl_epsilon_decay_steps": 400
    },
    "High_Density_Dynamic": {
        "num_ues": 30,
        "num_bss": 5,
        "lambda_ue": 30 / (1000*1000*1e-6),
        "lambda_bs": 5 / (1000*1000*1e-6),
        "placement_method": "PPP",
        "ue_speed_mps": 8.0,
        "num_total_rbs": 50,
        "shadowing_std_dev_db": 6.0,
        "rl_learning_rate": 0.0001,
        "rl_batch_size": 64,
        "rl_replay_buffer_size": 10000,
        "rl_target_update_freq": 10,
        "rl_gamma": 0.99,
        "rl_hidden_units": 128,
        "rl_num_hidden_layers": 3,
        "rl_dropout_rate": 0.2,
        "rl_epsilon_decay_steps": 400
    },
    # ... update other scenarios similarly ...
}

# Increase number of seeds for better statistical significance
NUM_SEEDS_PER_RUN = 5  # Increased from 3

# ... rest of the existing code ... 