class SimParams:
    def __init__(self):
        # General
        self.placement_method = "Uniform Random"
        self.num_ues = 10
        self.num_bss = 3
        self.lambda_bs = 3 / (1000 * 1000 * 1e-6)  # Per km^2
        self.lambda_ue = 10 / (1000 * 1000 * 1e-6) # Per km^2
        self.sim_area_x = 1000
        self.sim_area_y = 1000
        self.time_step_duration = 0.2
        self.total_sim_steps = 50

        # BS Parameters
        self.bs_tx_power_dbm = 38.0
        self.bs_link_beam_gain_db = 10.0
        self.bs_access_beam_gain_db = 3.0

        # UE Parameters
        self.ue_speed_mps = 5.0
        self.ue_noise_figure_db = 7.0
        self.target_ue_throughput_mbps = 1.0
        self.max_rbs_per_ue = 4
        self.max_rbs_per_ue_per_bs = 2

        # RB Parameters
        self.num_total_rbs = 20
        self.rb_bandwidth_mhz = 0.5
        self.rb_bandwidth_hz = self.rb_bandwidth_mhz * 1e6

        # Channel Model
        self.path_loss_exponent = 3.7
        self.ref_dist_m = 1.0
        self.ref_loss_db = 32.0
        self.shadowing_std_dev_db = 4.0

        # Handover Parameters (for Baseline Agent)
        self.ho_hysteresis_db = 3.0
        self.ho_time_to_trigger_s = 0.4
        self.min_rsrp_for_acq_dbm = -115.0

        # RL Parameters
        self.rl_agent_type = "Baseline"
        self.rl_gamma = 0.99  # Increased from 0.95 for better long-term planning
        self.rl_learning_rate = 0.0005  # Reduced from 0.001 for more stable training
        self.rl_epsilon_start = 1.0
        self.rl_epsilon_end = 0.05
        self.rl_epsilon_decay_steps = 500  # Increased from 200 for slower exploration decay
        self.rl_batch_size = 32  # Increased from 16 for better gradient estimates
        self.rl_target_update_freq = 5
        self.rl_replay_buffer_size = 50000  # Increased from 1000 for better experience diversity
        self.rl_n_step_sarsa = 3

        # DQN-Specific Architecture Parameters
        self.rl_hidden_units = 128  # Increased from 64 for more capacity
        self.rl_num_hidden_layers = 3  # Increased from 2 for deeper network
        self.rl_dropout_rate = 0.2  # Increased from 0.0 for better regularization
        self.rl_use_soft_updates = True  # Changed to True for more stable training
        self.rl_tau = 0.005  # Tau for soft updates

        # To be calculated by simulation during setup (not direct user inputs)
        self.ho_time_to_trigger_steps = 0
        self.num_ues_actual = self.num_ues
        self.num_bss_actual = self.num_bss
        self.channel_model = None

    def to_dict(self):
        """
        Converts the SimParams object to a dictionary, making it suitable for
        serialization (e.g., to JSON for saving).
        """
        param_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith("_") and not callable(
                getattr(self, attr_name)
            ):
                value = getattr(self, attr_name)
                if attr_name == "channel_model" and value is not None:
                    continue
                param_dict[attr_name] = value
        return param_dict