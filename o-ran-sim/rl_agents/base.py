import random
from sim_core.params import SimParams

class RLAgentBase:
    def __init__(self, params: SimParams, num_bss, num_ues, logger_func):
        self.params = params
        self.num_bss = num_bss
        self.num_ues = num_ues
        self.logger = logger_func
        self.current_epsilon = params.rl_epsilon_start
        self.total_steps_taken = 0

    def get_action(self, state, ue_id, available_bs_ids=None):
        raise NotImplementedError

    def train(self, state, action, reward, next_state, done, ue_id=None):
        pass # pass since not all agents might train this way (e.g. Baseline)

    def update_epsilon(self):
        self.total_steps_taken += 1
        if self.params.rl_epsilon_decay_steps > 0:
            fraction = min(
                1.0, self.total_steps_taken / self.params.rl_epsilon_decay_steps
            )
            self.current_epsilon = self.params.rl_epsilon_start + fraction * (
                self.params.rl_epsilon_end - self.params.rl_epsilon_start
            )
        else:
            self.current_epsilon = self.params.rl_epsilon_end

    def log(self, message):
        if self.logger:
            self.logger(f"RL_AGENT: {message}")