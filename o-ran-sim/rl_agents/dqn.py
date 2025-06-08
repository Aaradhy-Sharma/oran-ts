import collections
import numpy as np
import random
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

from sim_core.params import SimParams
from sim_core.entities import UserEquipment
from sim_core.resource import ResourceBlockPool
from rl_agents.base import RLAgentBase

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. DQNAgent cannot be used.")

class LossTracker(Callback):
    """Custom callback to track loss during training."""
    def __init__(self):
        super().__init__()
        self.losses = []
        self.episode_losses = []
        self.current_episode_losses = []
    
    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            loss = logs.get('loss')
            if loss is not None:
                self.losses.append(loss)
                self.current_episode_losses.append(loss)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.current_episode_losses:
            self.episode_losses.append(np.mean(self.current_episode_losses))
            self.current_episode_losses = []

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.memory)]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class DQNAgent(RLAgentBase):
    def __init__(
        self, params: SimParams, state_size, action_size, num_bss_total, logger_func
    ):
        super().__init__(params, num_bss_total, params.num_ues_actual, logger_func)
        self.state_size = state_size
        self.action_size = action_size
        self.num_bss_total = num_bss_total
        self.num_rb_categories = 3
        self.loss_tracker = LossTracker()
        self.current_loss = 0.0
        self.episode_losses = []

        # Initialize prioritized replay buffer with larger capacity
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=params.rl_replay_buffer_size,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=100000
        )

        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for DQNAgent. Please install TensorFlow to use this agent."
            )

        # Initialize optimizer with gradient clipping and learning rate schedule
        initial_learning_rate = params.rl_learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        self.optimizer = Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.log(
            f"DQNAgent initialized. State: {state_size}, Action: {action_size}, "
            f"Arch: {params.rl_num_hidden_layers}x{params.rl_hidden_units}"
        )

    def _build_model(self):
        # Create input layer with proper shape
        inputs = Input(shape=(self.state_size,))
        x = inputs
        
        # First hidden layer with batch normalization and residual connection
        x = Dense(256, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        residual = x
        
        # Additional hidden layers with residual connections
        for i in range(self.params.rl_num_hidden_layers - 1):
            x = Dense(256, kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            # Add residual connection every other layer
            if i % 2 == 0:
                x = tf.keras.layers.Add()([x, residual])
                residual = x
            
            if self.params.rl_dropout_rate > 0:
                x = Dropout(self.params.rl_dropout_rate)(x)

        # Final hidden layer
        x = Dense(128, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        # Output layer with proper initialization
        outputs = Dense(
            self.action_size,
            activation="linear",
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=self.optimizer,
            loss=self._huber_loss
        )
        return model

    def _huber_loss(self, y_true, y_pred):
        return tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)

    def remember(self, state, action_idx, reward, next_state, done):
        # Clip reward to prevent instability
        reward = np.clip(reward, -1.0, 1.0)
        self.replay_buffer.push(state, action_idx, reward, next_state, done)

    def get_action(self, state_array, ue_id=None, available_bs_ids=None):
        # Normalize state input
        state_array = state_array / (np.linalg.norm(state_array) + 1e-6)
        
        if random.random() < self.current_epsilon:
            return random.randrange(self.action_size)
        state_tensor = tf.convert_to_tensor(
            state_array.reshape(1, -1), dtype=tf.float32
        )
        q_values = self.model.predict(state_tensor, verbose=0)
        return np.argmax(q_values[0])

    def train(
        self, state=None, action=None, reward=None, next_state=None, done=None, ue_id=None
    ):
        # Delayed training start - wait until buffer is warm
        if len(self.replay_buffer.memory) < self.params.rl_batch_size:
            return

        # Sample from prioritized replay buffer
        minibatch, indices, weights = self.replay_buffer.sample(self.params.rl_batch_size)

        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        # Normalize states
        states = states / (np.linalg.norm(states, axis=1, keepdims=True) + 1e-6)
        next_states = next_states / (np.linalg.norm(next_states, axis=1, keepdims=True) + 1e-6)

        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)

        # Get current Q-values and next Q-values from target network
        current_q_values = self.model.predict(states_tensor, verbose=0)
        next_q_values_target = self.target_model.predict(next_states_tensor, verbose=0)

        # Calculate target Q-values with reward shaping and double Q-learning
        targets = np.copy(current_q_values)
        for i in range(self.params.rl_batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                # Double Q-learning: use online network to select action, target network to evaluate
                next_actions = np.argmax(self.model.predict(next_states_tensor[i:i+1], verbose=0)[0])
                next_q_value = next_q_values_target[i, next_actions]
                
                # Add reward shaping based on state features
                satisfaction_bonus = 0.1 if next_states[i][0] > 0.5 else 0  # Bonus for satisfaction
                load_penalty = -0.05 * (next_states[i][1] + next_states[i][2])  # Penalty for high load
                rsrp_bonus = 0.05 * next_states[i][3]  # Bonus for good RSRP difference
                
                shaped_reward = rewards[i] + satisfaction_bonus + load_penalty + rsrp_bonus
                # Clip shaped reward
                shaped_reward = np.clip(shaped_reward, -1.0, 1.0)
                targets[i, actions[i]] = (
                    shaped_reward + self.params.rl_gamma * next_q_value
                )

        # Calculate TD errors for priority update
        td_errors = np.abs(targets - current_q_values).mean(axis=1)
        self.replay_buffer.update_priorities(indices, td_errors + 1e-6)

        # Train the model with importance sampling weights
        history = self.model.fit(
            states_tensor,
            targets,
            sample_weight=weights,
            epochs=1,
            verbose=0,
            callbacks=[self.loss_tracker]
        )
        
        # Update current loss and episode losses
        self.current_loss = history.history['loss'][0]
        self.episode_losses.append(self.current_loss)
        
        # Log loss for monitoring
        if hasattr(self, 'logger_func'):
            self.logger_func(f"DQN Loss: {self.current_loss:.4f}", level='debug')

    def update_target_model(self):
        if self.params.rl_use_soft_updates:
            q_weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            new_weights = []
            for i in range(len(target_weights)):
                new_weights.append(
                    self.params.rl_tau * q_weights[i]
                    + (1 - self.params.rl_tau) * target_weights[i]
                )
            self.target_model.set_weights(new_weights)
        else:  # Original hard update
            self.target_model.set_weights(self.model.get_weights())
            self.log("DQL Target model updated (hard update).")

    def get_dql_action_space_size(self):
        choices_for_bs = self.num_bss_total + 1
        return (
            choices_for_bs
            * choices_for_bs
            * self.num_rb_categories
            * self.num_rb_categories
        )

    def map_action_idx_to_ue_config(self, action_idx):
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

        target_bs1_actual_idx = (
            bs1_choice_idx if bs1_choice_idx < self.num_bss_total else -1
        )
        target_bs2_actual_idx = (
            bs2_choice_idx if bs2_choice_idx < self.num_bss_total else -1
        )

        def map_cat_to_rbs(cat_idx):
            if cat_idx == 0:
                return 0
            max_rbs = max(1, self.params.max_rbs_per_ue_per_bs)
            if cat_idx == 1:
                return math.floor(max_rbs / 2)
            return max_rbs

        # Calculate number of RBs to request for each BS
        num_rbs_bs1 = map_cat_to_rbs(rbs1_cat_idx)
        num_rbs_bs2 = map_cat_to_rbs(rbs2_cat_idx)

        # Handle same BS case
        if target_bs1_actual_idx != -1 and target_bs1_actual_idx == target_bs2_actual_idx:
            target_bs2_actual_idx = -1
            num_rbs_bs2 = 0

        # Ensure total RBs don't exceed max per UE
        if num_rbs_bs1 + num_rbs_bs2 > self.params.max_rbs_per_ue:
            if num_rbs_bs1 >= self.params.max_rbs_per_ue:
                num_rbs_bs1 = self.params.max_rbs_per_ue
                num_rbs_bs2 = 0
            else:
                num_rbs_bs2 = self.params.max_rbs_per_ue - num_rbs_bs1

        # Handle no BS cases
        if target_bs1_actual_idx == -1:
            num_rbs_bs1 = 0
        if target_bs2_actual_idx == -1:
            num_rbs_bs2 = 0

        return target_bs1_actual_idx, target_bs2_actual_idx, num_rbs_bs1, num_rbs_bs2

    def get_dql_state_size(self, num_top_k_rsrp=3):
        # Simplified state representation focusing on key metrics
        return 5  # [satisfaction, bs1_load, bs2_load, rsrp_diff, current_rate_ratio]

    def get_dql_state_for_ue(
        self, ue: UserEquipment, bss_map, rb_pool: ResourceBlockPool, num_top_k_rsrp=3
    ):
        # Satisfaction state (0: not satisfied, 1: satisfied)
        is_satisfied = ue.current_total_rate_mbps >= self.params.target_ue_throughput_mbps
        satisfaction_state = 1.0 if is_satisfied else 0.0

        # Load states for both serving BSs
        bs1_load = ue.serving_bs_1.load_factor_metric if ue.serving_bs_1 else 0.0
        bs2_load = ue.serving_bs_2.load_factor_metric if ue.serving_bs_2 else 0.0

        # RSRP difference state with more granular levels
        rsrp_diff = 0.0
        if ue.serving_bs_1 and ue.best_bs_candidates:
            rsrp_serv = ue.measurements.get(ue.serving_bs_1.id, -np.inf)
            best_neigh_rsrp = -np.inf
            for cand_id, cand_rsrp in ue.best_bs_candidates:
                if cand_id == ue.serving_bs_1.id:
                    continue
                if cand_rsrp > best_neigh_rsrp:
                    best_neigh_rsrp = cand_rsrp
            
            if best_neigh_rsrp > rsrp_serv + self.params.ho_hysteresis_db + 6:
                rsrp_diff = 1.0
            elif best_neigh_rsrp > rsrp_serv + self.params.ho_hysteresis_db + 3:
                rsrp_diff = 0.75
            elif best_neigh_rsrp > rsrp_serv + self.params.ho_hysteresis_db:
                rsrp_diff = 0.5
            elif best_neigh_rsrp > rsrp_serv:
                rsrp_diff = 0.25

        # Current rate ratio (normalized to target) with more granular levels
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

        return state