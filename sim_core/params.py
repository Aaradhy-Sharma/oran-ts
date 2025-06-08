class SimParams:
    def __init__(self):
        # ... keep existing general parameters ...

        # RL Parameters - Updated for better DQN performance
        self.rl_agent_type = "Baseline"
        self.rl_gamma = 0.99  # Increased from 0.95 for better long-term reward consideration
        self.rl_learning_rate = 0.0001  # Reduced from 0.001 for more stable learning
        self.rl_epsilon_start = 1.0
        self.rl_epsilon_end = 0.1  # Increased from 0.05 for better exploration
        self.rl_epsilon_decay_steps = 1000  # Increased from 200 for longer exploration
        self.rl_batch_size = 64  # Increased from 16 for better gradient estimates
        self.rl_target_update_freq = 10  # Increased from 5 for more stable learning
        self.rl_replay_buffer_size = 10000  # Increased from 1000 for more diverse experiences
        
        # New DQN-specific parameters
        self.rl_hidden_units = 128  # Increased from 64
        self.rl_num_hidden_layers = 3  # Increased from 2
        self.rl_dropout_rate = 0.2  # New parameter for regularization
        
        # ... keep rest of the existing parameters ... 