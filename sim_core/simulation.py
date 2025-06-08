def _initialize_rl_agent(self):
    agent_type = self.params.rl_agent_type
    num_bss_act = self.params.num_bss_actual
    num_ues_act = self.params.num_ues_actual

    if agent_type == "Baseline":
        return BaselineAgent(self.params, num_bss_act, num_ues_act, self.log_message, self.bss, self.rb_pool)
    elif agent_type == "TabularQLearning":
        return TabularQLearningAgent(self.params, num_bss_act, num_ues_act, self.log_message)
    elif agent_type == "SARSA":
        return SARSAAgent(self.params, num_bss_act, num_ues_act, self.log_message)
    elif agent_type == "ExpectedSARSA":
        return ExpectedSARSAAgent(self.params, num_bss_act, num_ues_act, self.log_message)
    elif agent_type == "NStepSARSA":
        n_step_val = getattr(self.params, "rl_n_step_sarsa", 3)
        return NStepSARSAAgent(self.params, num_bss_act, num_ues_act, self.log_message, n_step=n_step_val)
    elif agent_type == "DQN":
        if not TF_AVAILABLE:
            self.log_message("ERROR: TensorFlow not available for DQN. Falling back to Baseline.")
            self.params.rl_agent_type = "Baseline"
            return BaselineAgent(self.params, num_bss_act, num_ues_act, self.log_message, self.bss, self.rb_pool)
        temp_dqn_for_sizes = DQNAgent(self.params, 10, 10, num_bss_act, self.log_message)
        state_size = temp_dqn_for_sizes.get_dqn_state_size()
        action_size = temp_dqn_for_sizes.get_dqn_action_space_size()
        del temp_dqn_for_sizes
        return DQNAgent(self.params, state_size, action_size, num_bss_act, self.log_message)
    else:
        self.log_message(f"Unknown RL agent type: {agent_type}. Using Baseline.")
        self.params.rl_agent_type = "Baseline"
        return BaselineAgent(self.params, num_bss_act, num_ues_act, self.log_message, self.bss, self.rb_pool) 