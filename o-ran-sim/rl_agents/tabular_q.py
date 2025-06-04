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
        # State: (is_satisfied_cat, serving_bs1_load_cat, best_neighbor_rsrp_cat_vs_serv)
        # Action: (0: stay, 1: switch_to_best_neighbor, 2: try_dual_connect_best_neighbor)
        self.q_table = collections.defaultdict(lambda: np.zeros(3)) # 3 actions
        self.log("TabularQLearningAgent initialized.")

    def _get_state_representation(self, ue: UserEquipment, bss_map):
        # Very coarse state representation
        is_satisfied = ue.current_total_rate_mbps >= self.params.target_ue_throughput_mbps
        satisfied_cat = 1 if is_satisfied else 0

        bs1_load_cat = 0 # low
        if ue.serving_bs_1:
            load = ue.serving_bs_1.load_factor_metric
            if load > 0.8:
                bs1_load_cat = 2 # high
            elif load > 0.4:
                bs1_load_cat = 1 # medium

        rsrp_diff_cat = 0 # neighbor weaker or much stronger
        if ue.serving_bs_1 and ue.best_bs_candidates:
            rsrp_serv = ue.measurements.get(ue.serving_bs_1.id, -np.inf)
            best_neigh_rsrp = -np.inf
            # Find best non-serving candidate
            for cand_id, cand_rsrp in ue.best_bs_candidates:
                if cand_id == ue.serving_bs_1.id:
                    continue
                if cand_rsrp > best_neigh_rsrp:
                    best_neigh_rsrp = cand_rsrp
            
            if (
                best_neigh_rsrp > rsrp_serv + self.params.ho_hysteresis_db + 3
            ): # Significantly stronger
                rsrp_diff_cat = 2
            elif (
                best_neigh_rsrp > rsrp_serv + self.params.ho_hysteresis_db
            ): # Stronger
                rsrp_diff_cat = 1

        return (satisfied_cat, bs1_load_cat, rsrp_diff_cat)

    def get_action(self, state_tuple, ue_id, available_bs_ids=None):
        if random.random() < self.current_epsilon:
            return random.choice([0, 1, 2]) # Explore
        return np.argmax(self.q_table[state_tuple]) # Exploit

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