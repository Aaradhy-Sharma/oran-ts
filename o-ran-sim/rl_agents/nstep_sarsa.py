import collections
import numpy as np
import random
from sim_core.params import SimParams
from sim_core.entities import UserEquipment
from rl_agents.base import RLAgentBase

class NStepSARSAAgent(RLAgentBase):
    def __init__(self, params: SimParams, num_bss, num_ues, logger_func, n_step=3):
        super().__init__(params, num_bss, num_ues, logger_func)
        self.q_table = collections.defaultdict(lambda: np.zeros(3))
        self.n_step = n_step
        self.ue_trajectories = {
            f"UE{i}": collections.deque(maxlen=self.n_step)
            for i in range(params.num_ues_actual)
        }
        self.log(f"NStepSARSAAgent initialized with N={self.n_step}.")

    def _get_state_representation(self, ue: UserEquipment, bss_map):
        # Same as Tabular Q-Learning Agent << 
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
            action_idx = random.choice([0, 1, 2])
        else:
            action_idx = np.argmax(self.q_table[state_tuple])
        return action_idx

    def record_experience_and_train(self, ue_id, state_tuple, action_idx, reward, done):
        # Add current experience (S_t, A_t, R_t+1) to trajectory
        self.ue_trajectories[ue_id].append((state_tuple, action_idx, reward))

        # We can only perform an update when we have at least N steps in the trajectory
        # and we know S_t+N (which is current state_tuple) and A_t+N (which will be chosen next)
        # For simplicity, this implementation uses the state S_t+N from *this* step
        # and assumes A_t+N is chosen after this step by get_action.
        # This means the update is for S_t-N.

        if len(self.ue_trajectories[ue_id]) == self.n_step + 1: # We have S_tau...S_tau+N, A_tau...A_tau+N-1, R_tau+1...R_tau+N
            # Remove the oldest experience (S_tau, A_tau, R_tau+1) to process it
            s_tau, a_tau, _ = self.ue_trajectories[ue_id].popleft() # Pop oldest for update

            G = 0 # N-step return
            # Sum discounted rewards from R_tau+1 to R_tau+N
            for i in range(self.n_step):
                _s_i, _a_i, r_i = self.ue_trajectories[ue_id][i]
                G += (self.params.rl_gamma**i) * r_i

            # Get S_tau+N and A_tau+N (which is S_t, A_t chosen from S_t from the perspective of _get_action)
            s_tau_plus_n = self.ue_trajectories[ue_id][-1][0] # The current state S_t
            a_tau_plus_n = self.ue_trajectories[ue_id][-1][1] # The action A_t chosen from S_t

            if not done:
                G += (self.params.rl_gamma**self.n_step) * self.q_table[s_tau_plus_n][a_tau_plus_n]

            old_q_value = self.q_table[s_tau][a_tau]
            self.q_table[s_tau][a_tau] = old_q_value + self.params.rl_learning_rate * (
                G - old_q_value
            )
        elif done and len(self.ue_trajectories[ue_id]) > 0: # If episode ends early, flush remaining experiences
            while len(self.ue_trajectories[ue_id]) > 0:
                s_tau, a_tau, _ = self.ue_trajectories[ue_id].popleft()
                G = 0
                for i in range(len(self.ue_trajectories[ue_id])):
                    _s_i, _a_i, r_i = self.ue_trajectories[ue_id][i]
                    G += (self.params.rl_gamma**i) * r_i

                old_q_value = self.q_table[s_tau][a_tau]
                self.q_table[s_tau][a_tau] = old_q_value + self.params.rl_learning_rate * (
                    G - old_q_value
                )

    def train(self, state, action, reward, next_state, done, ue_id=None):
        # This method is not directly used by NStepSARSA.
        pass