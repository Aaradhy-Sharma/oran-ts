import collections
import numpy as np
import random
import math
from sim_core.params import SimParams
from sim_core.entities import UserEquipment
from sim_core.resource import ResourceBlockPool
from rl_agents.base import RLAgentBase
from sim_core.constants import TF_AVAILABLE as TF_AVAILABLE_GLOBAL

# --- RL Agent Related Imports (TensorFlow for DQL) ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE_GLOBAL = True # Update global flag
except ImportError:
    TF_AVAILABLE_GLOBAL = False
    print("TensorFlow not available. DQL agent will not work.")


class DQNAgent(RLAgentBase):
    def __init__(self, params: SimParams, state_size, action_size, num_bss_total, logger_func):
        super().__init__(params, num_bss_total, params.num_ues_actual, logger_func)
        self.state_size = state_size
        self.action_size = action_size
        self.num_bss_total = num_bss_total
        self.num_rb_categories = 3 # 0, 1/2 max, max

        self.replay_buffer = collections.deque(maxlen=params.rl_replay_buffer_size)

        if TF_AVAILABLE_GLOBAL:
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.target_model.set_weights(self.model.get_weights())
        else:
            self.log("ERROR: TensorFlow not found. DQL Agent cannot be created.")
            raise ImportError("TensorFlow is required for DQNAgent.")

        self.log(
            f"DQNAgent initialized. State size: {state_size}, Action size: {action_size}"
        )

    def _build_model(self):
        model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(self.state_size,)),
                Dense(64, activation="relu"),
                Dense(self.action_size, activation="linear"), # Q-values for each action
            ]
        )
        model.compile(optimizer=Adam(learning_rate=self.params.rl_learning_rate), loss="mse")
        return model

    def remember(self, state, action_idx, reward, next_state, done):
        self.replay_buffer.append((state, action_idx, reward, next_state, done))

    def get_action(self, state_array, ue_id=None, available_bs_ids=None):
        if random.random() < self.current_epsilon:
            return random.randrange(self.action_size) # Explore
        state_tensor = tf.convert_to_tensor(state_array.reshape(1, -1), dtype=tf.float32)
        q_values = self.model.predict(state_tensor, verbose=0)
        return np.argmax(q_values[0]) # Exploit: action index with highest Q-value

    def train(self, state=None, action=None, reward=None, next_state=None, done=None, ue_id=None):
        if len(self.replay_buffer) < self.params.rl_batch_size:
            return

        minibatch = random.sample(self.replay_buffer, self.params.rl_batch_size)

        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)

        current_q_values = self.model.predict(states_tensor, verbose=0)
        next_q_values_target = self.target_model.predict(next_states_tensor, verbose=0)

        targets = np.copy(current_q_values)

        for i in range(self.params.rl_batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.params.rl_gamma * np.max(
                    next_q_values_target[i]
                )

        self.model.fit(states_tensor, targets, epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.log("DQL Target model updated.")

    def get_dql_action_space_size(self):
        # Action space: (BS1_choice, BS2_choice, RBS1_cat, RBS2_cat)
        # BS_choice: 0..N-1 (actual BS index) or N (for None/no connection)
        # So, N_bss_total + 1 choices for each BS
        choices_for_bs = self.num_bss_total + 1
        return (
            choices_for_bs
            * choices_for_bs
            * self.num_rb_categories
            * self.num_rb_categories
        )

    def map_action_idx_to_ue_config(self, action_idx):
        # Inverse of the linearization for action_idx
        choices_for_bs = self.num_bss_total + 1

        actions_per_bs1_choice = (
            choices_for_bs * self.num_rb_categories * self.num_rb_categories
        )
        actions_per_bs2_choice = self.num_rb_categories * self.num_rb_categories

        bs1_choice_idx = action_idx // actions_per_bs1_choice
        remainder = action_idx % actions_per_bs1_choice

        bs2_choice_idx = remainder // actions_per_bs2_choice
        remainder = remainder % actions_per_bs2_choice

        rbs1_cat_idx = remainder // self.num_rb_categories
        rbs2_cat_idx = remainder % self.num_rb_categories

        # Convert choice indices to actual BS indices (-1 for None)
        target_bs1_actual_idx = bs1_choice_idx - 1 if bs1_choice_idx < choices_for_bs else -1
        target_bs2_actual_idx = bs2_choice_idx - 1 if bs2_choice_idx < choices_for_bs else -1

        def map_cat_to_rbs(cat_idx):
            if cat_idx == 0: return 0
            max_rbs = max(1, self.params.max_rbs_per_ue_per_bs)
            if cat_idx == 1: return math.floor(max_rbs / 2)
            return max_rbs # cat_idx == 2

        num_rbs_bs1 = map_cat_to_rbs(rbs1_cat_idx)
        num_rbs_bs2 = map_cat_to_rbs(rbs2_cat_idx)

        # Post-process for invalid combinations or resource caps
        # 1. Prevent connecting to same BS for both primary and secondary
        if target_bs1_actual_idx != -1 and target_bs1_actual_idx == target_bs2_actual_idx:
            target_bs2_actual_idx = -1
            num_rbs_bs2 = 0

        # 2. Cap total RBs if combined exceeds max_rbs_per_ue
        if num_rbs_bs1 + num_rbs_bs2 > self.params.max_rbs_per_ue:
            # Prioritize BS1 RBs, then fill with BS2 RBs up to total cap
            if num_rbs_bs1 >= self.params.max_rbs_per_ue:
                num_rbs_bs1 = self.params.max_rbs_per_ue
                num_rbs_bs2 = 0
            else:
                num_rbs_bs2 = self.params.max_rbs_per_ue - num_rbs_bs1
        
        # 3. If a BS is "None" (-1), ensure its RBs are 0
        if target_bs1_actual_idx == -1:
            num_rbs_bs1 = 0
        if target_bs2_actual_idx == -1:
            num_rbs_bs2 = 0

        return target_bs1_actual_idx, target_bs2_actual_idx, num_rbs_bs1, num_rbs_bs2

    def get_dql_state_size(self, num_top_k_rsrp=3):
        # State: top K RSRPs, current rate, RBs from BS1, RBs from BS2, Load BS1, Load BS2
        ue_specific_size = num_top_k_rsrp + 1 + 1 + 1 + 1 + 1
        return ue_specific_size

    def get_dql_state_for_ue(self, ue: UserEquipment, bss_map, rb_pool: ResourceBlockPool, num_top_k_rsrp=3):
        state = []
        # Normalized top K RSRPs
        sorted_measurements = sorted(
            ue.measurements.items(), key=lambda item: item[1], reverse=True
        )
        for i in range(num_top_k_rsrp):
            if i < len(sorted_measurements):
                # Normalize RSRP from [-120, -70] dBm to [0, 1]
                rsrp_norm = (sorted_measurements[i][1] - (-120)) / ((-70) - (-120))
                state.append(np.clip(rsrp_norm, 0, 1))
            else:
                state.append(0.0) # Pad with 0 if fewer than K candidates

        # Normalized current total rate
        state.append(np.clip(ue.current_total_rate_mbps / self.params.target_ue_throughput_mbps, 0, 2)) # Cap at 2x target

        # Normalized RBs from BS1 and BS2
        state.append(len(ue.rbs_from_bs1) / max(1, self.params.max_rbs_per_ue_per_bs))
        state.append(len(ue.rbs_from_bs2) / max(1, self.params.max_rbs_per_ue_per_bs))
        
        # Load factors of serving BSs (normalized implicitly 0-1)
        state.append(ue.serving_bs_1.load_factor_metric if ue.serving_bs_1 else 0.0)
        state.append(ue.serving_bs_2.load_factor_metric if ue.serving_bs_2 else 0.0)

        return np.array(state, dtype=np.float32)