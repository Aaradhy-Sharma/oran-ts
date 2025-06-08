import collections
import numpy as np
import random
from sim_core.params import SimParams
from sim_core.entities import UserEquipment
from rl_agents.base import RLAgentBase

# type: class TabularQLearningAgent(RLAgentBase)
class TabularQLearningAgent(RLAgentBase):
    def __init__(self, params: SimParams, num_bss, num_ues, logger_func):
        super().__init__(params, num_bss, num_ues, logger_func)
        # State: (satisfaction, bs1_load, bs2_load, rsrp_diff, rate_ratio)
        # Each component is discretized into 10 bins
        self.num_bins = 10
        self.q_table = collections.defaultdict(lambda: np.zeros(3))  # 3 actions
        self.log("TabularQLearningAgent initialized with granular state space.")

    def _discretize_state(self, state_array):
        # Discretize each continuous state component into bins
        satisfaction_bin = min(int(state_array[0] * self.num_bins), self.num_bins - 1)
        bs1_load_bin = min(int(state_array[1] * self.num_bins), self.num_bins - 1)
        bs2_load_bin = min(int(state_array[2] * self.num_bins), self.num_bins - 1)
        rsrp_diff_bin = min(int(state_array[3] * self.num_bins), self.num_bins - 1)
        rate_ratio_bin = min(int(state_array[4] * self.num_bins), self.num_bins - 1)
        
        return (satisfaction_bin, bs1_load_bin, bs2_load_bin, rsrp_diff_bin, rate_ratio_bin)

    def _get_state_representation(self, ue: UserEquipment, bss_map):
        # Satisfaction state (0: not satisfied, 1: satisfied)
        is_satisfied = ue.current_total_rate_mbps >= self.params.target_ue_throughput_mbps
        satisfaction_state = 1.0 if is_satisfied else 0.0

        # Load states for both serving BSs
        bs1_load = ue.serving_bs_1.load_factor_metric if ue.serving_bs_1 else 0.0
        bs2_load = ue.serving_bs_2.load_factor_metric if ue.serving_bs_2 else 0.0

        # RSRP difference state
        rsrp_diff = 0.0
        if ue.serving_bs_1 and ue.best_bs_candidates:
            rsrp_serv = ue.measurements.get(ue.serving_bs_1.id, -np.inf)
            best_neigh_rsrp = -np.inf
            for cand_id, cand_rsrp in ue.best_bs_candidates:
                if cand_id == ue.serving_bs_1.id:
                    continue
                if cand_rsrp > best_neigh_rsrp:
                    best_neigh_rsrp = cand_rsrp
            
            if best_neigh_rsrp > rsrp_serv + self.params.ho_hysteresis_db + 3:
                rsrp_diff = 1.0
            elif best_neigh_rsrp > rsrp_serv + self.params.ho_hysteresis_db:
                rsrp_diff = 0.5

        # Current rate ratio (normalized to target)
        rate_ratio = np.clip(
            ue.current_total_rate_mbps / self.params.target_ue_throughput_mbps,
            0.0, 2.0
        )

        state = np.array([
            satisfaction_state,
            bs1_load,
            bs2_load,
            rsrp_diff,
            rate_ratio
        ], dtype=np.float32)

        return self._discretize_state(state)

    def get_action(self, state_tuple, ue_id, available_bs_ids=None):
        if random.random() < self.current_epsilon:
            return random.choice([0, 1, 2])  # Explore
        return np.argmax(self.q_table[state_tuple])  # Exploit

    def train(self, state_tuple, action, reward, next_state_tuple, done, ue_id=None):
        old_value = self.q_table[state_tuple][action]
        if done:
            target = reward
        else:
            target = reward + self.params.rl_gamma * np.max(
                self.q_table[next_state_tuple]
            )

        new_value = old_value + self.params.rl_learning_rate * (target - old_value)
        self.q_table[state_tuple][action] = new_value