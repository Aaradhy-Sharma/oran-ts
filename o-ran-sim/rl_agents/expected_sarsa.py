import collections
import numpy as np
import random
from sim_core.params import SimParams
from sim_core.entities import UserEquipment
from rl_agents.base import RLAgentBase

class ExpectedSARSAAgent(RLAgentBase):
    def __init__(self, params: SimParams, num_bss, num_ues, logger_func):
        super().__init__(params, num_bss, num_ues, logger_func)
        self.q_table = collections.defaultdict(
            lambda: np.zeros(3)
        ) # (stay, switch_single, try_dual) << 
        self.num_actions = 3
        self.log("ExpectedSARSAAgent initialized.")

    def _get_state_representation(self, ue: UserEquipment, bss_map):
        # Same as Tabular Q-Learning Agent
        is_satisfied = ue.current_total_rate_mbps >= self.params.target_ue_throughput_mbps
        satisfied_cat = 1 if is_satisfied else 0
        bs1_load_cat = 0
        if ue.serving_bs_1:
            load = ue.serving_bs_1.load_factor_metric
            if load > 0.8:
                bs1_load_cat = 2
            elif load > 0.4:
                bs1_load_cat = 1
        rsrp_diff_cat = 0
        if ue.serving_bs_1 and ue.best_bs_candidates:
            rsrp_serv = ue.measurements.get(ue.serving_bs_1.id, -np.inf)
            best_neigh_rsrp = -np.inf
            for cand_id, cand_rsrp in ue.best_bs_candidates:
                if cand_id == ue.serving_bs_1.id:
                    continue
                if cand_rsrp > best_neigh_rsrp:
                    best_neigh_rsrp = cand_rsrp
            if (
                best_neigh_rsrp > rsrp_serv + self.params.ho_hysteresis_db + 3
            ):
                rsrp_diff_cat = 2
            elif best_neigh_rsrp > rsrp_serv + self.params.ho_hysteresis_db:
                rsrp_diff_cat = 1
        return (satisfied_cat, bs1_load_cat, rsrp_diff_cat)

    def get_action(self, state_tuple, ue_id, available_bs_ids=None):
        if random.random() < self.current_epsilon:
            return random.choice([0, 1, 2])
        return np.argmax(self.q_table[state_tuple])

    def train(self, s, a, r, s_prime, done, ue_id=None):
        old_value = self.q_table[s][a]

        if done:
            expected_next_q_value = 0
        else:
            q_values_s_prime = self.q_table[s_prime]
            greedy_action_s_prime = np.argmax(q_values_s_prime)

            expected_next_q_value = 0
            for action_idx_s_prime in range(self.num_actions):
                if action_idx_s_prime == greedy_action_s_prime:
                    policy_prob = (1 - self.current_epsilon) + (
                        self.current_epsilon / self.num_actions
                    )
                else:
                    policy_prob = self.current_epsilon / self.num_actions
                expected_next_q_value += policy_prob * q_values_s_prime[action_idx_s_prime]

        target = r + self.params.rl_gamma * expected_next_q_value
        new_value = old_value + self.params.rl_learning_rate * (target - old_value)
        self.q_table[s][a] = new_value