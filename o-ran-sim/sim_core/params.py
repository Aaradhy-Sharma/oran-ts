class SimParams:
    def __init__(self):
        # General
        self.placement_method = "Uniform Random"
        self.num_ues = 10
        self.num_bss = 3
        self.lambda_bs = 3 / (1000 * 1000 * 1e-6) # Per km^2
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
        self.rb_bandwidth_hz = self.rb_bandwidth_mhz * 1e6 # Will be calculated based on mhz

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
        self.rl_gamma = 0.95
        self.rl_learning_rate = 0.001
        self.rl_epsilon_start = 1.0
        self.rl_epsilon_end = 0.05
        self.rl_epsilon_decay_steps = 200
        self.rl_batch_size = 16
        self.rl_target_update_freq = 5
        self.rl_replay_buffer_size = 1000
        self.rl_n_step_sarsa = 3 # N-step for NStepSARSA

        # To be calculated by simulation during setup (not direct user inputs)
        self.ho_time_to_trigger_steps = 0
        self.num_ues_actual = self.num_ues # Will be updated by PPP if applicable
        self.num_bss_actual = self.num_bss # Will be updated by PPP if applicable
        self.channel_model = None # Will be linked from Simulation class

    def to_dict(self):
        """
        Converts the SimParams object to a dictionary, making it suitable for
        serialization (e.g., to JSON for saving).
        Excludes callable attributes and dunder methods.
        """
        param_dict = {}
        for attr_name in dir(self):
            # Exclude private/protected attributes and methods
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                value = getattr(self, attr_name)
                # Handle special cases if any objects are not directly JSON serializable
                # For example, if channel_model was an object, you might want to serialize its ID or skip it.
                if attr_name == "channel_model" and value is not None:
                    continue # Skip the object reference, it's not a parameter to save directly
                param_dict[attr_name] = value
        return param_dict