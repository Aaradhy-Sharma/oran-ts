"""
Fixed DQN Agent with improved stability and performance.

Key improvements over original DQN:
1. Global state normalization with running statistics
2. Simplified action space (27 actions instead of 72)
3. Better reward scaling without aggressive clipping
4. Optimized learning rate (no fast decay)
5. Improved exploration schedule
6. Better target network updates (soft updates)
7. Prioritized Experience Replay with Double DQN
8. Larger batch size for stability
"""

import collections
import numpy as np
import random
import math
from typing import Optional, Tuple, Dict, Any

from sim_core.params import SimParams
from sim_core.entities import UserEquipment
from sim_core.resource import ResourceBlockPool
from rl_agents.base import RLAgentBase

# Guard TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add, Activation
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


class ReplayBuffer:
    """Simple uniform replay buffer."""
    
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBufferFixed:
    """Prioritized experience replay buffer with fixed sampling."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """Sample batch with importance sampling weights."""
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32)
        )
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return self.size


class DQNAgentFixed(RLAgentBase):
    """
    Fixed DQN Agent with improved stability and performance.
    """
    
    def __init__(
        self, 
        params: SimParams, 
        state_size: int, 
        action_size: int,
        num_bss_total: int,
        logger_func
    ):
        super().__init__(params, num_bss_total, params.num_ues_actual, logger_func)
        
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for DQNAgentFixed. Please install TensorFlow to use this agent."
            )
        
        self.state_size = state_size
        self.num_bss_total = num_bss_total
        
        # Simplified action space: (bs1_choice, bs2_choice, rb_category)
        # bs1/bs2_choice: 0-(num_bss-1) or num_bss for "none"
        # rb_category: 0 (no RBs), 1 (half max), 2 (max)
        self.num_bs_choices = num_bss_total + 1  # Including "none"
        self.num_rb_categories = 3
        self.action_size = self.num_bs_choices * self.num_bs_choices * self.num_rb_categories
        
        # Network parameters
        self.hidden_layers = [256, 256, 128]
        self.activation = 'relu'
        self.dropout_rate = params.rl_dropout_rate if hasattr(params, 'rl_dropout_rate') else 0.0
        
        # Learning parameters - IMPROVED
        self.learning_rate = 0.0003  # Lower, more stable LR
        self.batch_size = 128  # Larger batch for stability
        self.gamma = params.rl_gamma
        
        # Experience replay - IMPROVED
        self.buffer_capacity = 50000  # Smaller buffer for better sample efficiency
        self.min_replay_size = 1000
        
        # Target network - IMPROVED
        self.target_update_freq = 500  # More frequent updates
        self.use_soft_update = True
        self.soft_tau = 0.01  # Faster soft updates
        
        # Prioritized replay
        self.use_per = True
        
        # Double DQN
        self.use_double_dqn = True
        
        # Reward processing - IMPROVED
        self.reward_scale = 0.1  # Scale down rewards
        self.reward_clip = None  # No clipping initially
        
        # State normalization - IMPROVED: Use running statistics
        self.running_mean = np.zeros(self.state_size, dtype=np.float32)
        self.running_var = np.ones(self.state_size, dtype=np.float32)
        self.normalization_count = 0
        self.normalization_momentum = 0.99
        
        # Build networks
        self.model = self._build_network("q_network")
        self.target_model = self._build_network("target_network")
        self._sync_target_network()
        
        # Optimizer - IMPROVED: Constant LR with gradient clipping
        self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        
        # Loss function
        self.loss_fn = tf.keras.losses.Huber(delta=1.0)
        
        # Replay buffer
        if self.use_per:
            self.replay_buffer = PrioritizedReplayBufferFixed(self.buffer_capacity)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        
        # Training state
        self.train_step_count = 0
        
        # Metrics
        self.episode_losses = []
        self.current_loss = 0.0
        
        self.log(
            f"DQNAgentFixed initialized. State: {state_size}, Action: {self.action_size}, "
            f"Architecture: {self.hidden_layers}, LR: {self.learning_rate}, Batch: {self.batch_size}"
        )
    
    def _build_network(self, name: str) -> Model:
        """Build Q-network with residual connections."""
        inputs = Input(shape=(self.state_size,), name=f"{name}_input")
        x = inputs
        
        # Input batch normalization
        x = BatchNormalization(name=f"{name}_input_bn")(x)
        
        # Residual blocks
        for i, units in enumerate(self.hidden_layers):
            residual = x
            
            # Main path
            x = Dense(units, kernel_initializer='he_normal', name=f"{name}_dense_{i}")(x)
            x = BatchNormalization(name=f"{name}_bn_{i}")(x)
            x = Activation(self.activation, name=f"{name}_act_{i}")(x)
            
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate, name=f"{name}_dropout_{i}")(x)
            
            # Residual connection (if dimensions match)
            if residual.shape[-1] == units:
                x = Add(name=f"{name}_residual_{i}")([x, residual])
        
        # Output layer
        outputs = Dense(
            self.action_size,
            activation='linear',
            kernel_initializer='he_normal',
            name=f"{name}_output"
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    def _update_running_stats(self, state_vec: np.ndarray):
        """Update running mean and variance for normalization."""
        self.normalization_count += 1
        
        # Update running mean and variance using exponential moving average
        self.running_mean = (
            self.normalization_momentum * self.running_mean +
            (1 - self.normalization_momentum) * state_vec
        )
        
        squared_diff = (state_vec - self.running_mean) ** 2
        self.running_var = (
            self.normalization_momentum * self.running_var +
            (1 - self.normalization_momentum) * squared_diff
        )
    
    def _normalize_state(self, state_vec: np.ndarray) -> np.ndarray:
        """Normalize state using running statistics."""
        # Update statistics during warmup
        if self.normalization_count < 1000:
            self._update_running_stats(state_vec)
        
        # Normalize
        std = np.sqrt(self.running_var + 1e-8)
        normalized = (state_vec - self.running_mean) / std
        
        # Clip to reasonable range
        normalized = np.clip(normalized, -5.0, 5.0)
        
        return normalized
    
    def _simplify_action_mapping(self, action_idx: int) -> Tuple[int, int, int, int]:
        """
        Simplified action mapping.
        
        Returns:
            (bs1_idx, bs2_idx, num_rbs_bs1, num_rbs_bs2)
        """
        # Decode factored action
        bs1_choice = action_idx // (self.num_bs_choices * self.num_rb_categories)
        remainder = action_idx % (self.num_bs_choices * self.num_rb_categories)
        bs2_choice = remainder // self.num_rb_categories
        rb_category = remainder % self.num_rb_categories
        
        # Map to actual BS indices (-1 means no connection)
        bs1_idx = bs1_choice if bs1_choice < self.num_bss_total else -1
        bs2_idx = bs2_choice if bs2_choice < self.num_bss_total else -1
        
        # Don't allow same BS twice
        if bs1_idx != -1 and bs1_idx == bs2_idx:
            bs2_idx = -1
        
        # Calculate RB allocation based on category
        max_rbs = max(1, self.params.max_rbs_per_ue_per_bs)
        if rb_category == 0:
            num_rbs_bs1 = 0
            num_rbs_bs2 = 0
        elif rb_category == 1:
            num_rbs_bs1 = math.floor(max_rbs / 2) if bs1_idx != -1 else 0
            num_rbs_bs2 = math.floor(max_rbs / 2) if bs2_idx != -1 else 0
        else:  # rb_category == 2
            num_rbs_bs1 = max_rbs if bs1_idx != -1 else 0
            num_rbs_bs2 = 0  # Simplified: max RBs to one BS only
        
        # Ensure no RBs if no BS
        if bs1_idx == -1:
            num_rbs_bs1 = 0
        if bs2_idx == -1:
            num_rbs_bs2 = 0
        
        return bs1_idx, bs2_idx, num_rbs_bs1, num_rbs_bs2
    
    def get_action(self, state_array, ue_id=None, available_bs_ids=None):
        """Select action using epsilon-greedy policy."""
        # Normalize state
        state_norm = self._normalize_state(state_array)
        
        # Epsilon-greedy
        if random.random() < self.current_epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            # Get Q-values
            state_batch = np.expand_dims(state_norm, axis=0)
            q_values = self.model.predict(state_batch, verbose=0)[0]
            action_idx = np.argmax(q_values)
        
        return action_idx
    
    def remember(self, state, action_idx, reward, next_state, done):
        """Store experience in replay buffer."""
        # Normalize states
        state_norm = self._normalize_state(state)
        next_state_norm = self._normalize_state(next_state)
        
        # Scale reward
        scaled_reward = reward * self.reward_scale
        
        # Clip reward if specified
        if self.reward_clip is not None:
            scaled_reward = np.clip(scaled_reward, -self.reward_clip, self.reward_clip)
        
        # Store transition
        self.replay_buffer.push(state_norm, action_idx, scaled_reward, next_state_norm, done)
    
    def train(self, state=None, action=None, reward=None, next_state=None, done=None, ue_id=None):
        """Perform one training step."""
        # Check if buffer is large enough
        if len(self.replay_buffer) < self.min_replay_size:
            return
        
        # Sample batch
        if self.use_per:
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights = np.ones(self.batch_size, dtype=np.float32)
            indices = None
        
        # Convert to tensors
        states = tf.constant(states, dtype=tf.float32)
        actions = tf.constant(actions, dtype=tf.int32)
        rewards = tf.constant(rewards, dtype=tf.float32)
        next_states = tf.constant(next_states, dtype=tf.float32)
        dones = tf.constant(dones, dtype=tf.float32)
        weights = tf.constant(weights, dtype=tf.float32)
        
        # Compute loss and gradients
        with tf.GradientTape() as tape:
            loss, td_errors = self._compute_loss(states, actions, rewards, next_states, dones, weights)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update priorities if using PER
        if self.use_per and indices is not None:
            priorities = td_errors.numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Update target network
        self.train_step_count += 1
        if self.use_soft_update:
            self._soft_update_target_network()
        elif self.train_step_count % self.target_update_freq == 0:
            self._sync_target_network()
        
        # Record metrics
        self.current_loss = float(loss.numpy())
        self.episode_losses.append(self.current_loss)
    
    @tf.function
    def _compute_loss(self, states, actions, rewards, next_states, dones, weights):
        """Compute loss with Double DQN."""
        # Current Q-values
        q_values = self.model(states, training=True)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1))
        
        # Next Q-values
        if self.use_double_dqn:
            # Double DQN: use online network to select action, target network to evaluate
            next_q_online = self.model(next_states, training=False)
            next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
            next_q_target = self.target_model(next_states, training=False)
            next_q_values = tf.gather_nd(
                next_q_target,
                tf.stack([tf.range(tf.shape(next_actions)[0]), next_actions], axis=1)
            )
        else:
            # Standard DQN
            next_q_values = tf.reduce_max(self.target_model(next_states, training=False), axis=1)
        
        # Compute targets
        targets = rewards + self.gamma * next_q_values * (1.0 - dones)
        targets = tf.stop_gradient(targets)
        
        # Compute TD errors for PER
        td_errors = tf.abs(targets - q_values)
        
        # Compute weighted loss
        loss = self.loss_fn(targets, q_values)
        weighted_loss = tf.reduce_mean(loss * weights)
        
        return weighted_loss, td_errors
    
    def _sync_target_network(self):
        """Hard update: copy weights from Q-network to target network."""
        self.target_model.set_weights(self.model.get_weights())
        self.log("DQN Fixed: Target model updated (hard update).")
    
    def _soft_update_target_network(self):
        """Soft update: slowly blend target network weights towards Q-network."""
        for target_var, source_var in zip(
            self.target_model.trainable_variables,
            self.model.trainable_variables
        ):
            target_var.assign(
                self.soft_tau * source_var + (1.0 - self.soft_tau) * target_var
            )
    
    def update_target_model(self):
        """Public method to update target network (for compatibility)."""
        if self.use_soft_update:
            self._soft_update_target_network()
        else:
            self._sync_target_network()
    
    def map_action_idx_to_ue_config(self, action_idx):
        """Map action index to UE configuration."""
        return self._simplify_action_mapping(action_idx)
    
    def get_dql_action_space_size(self):
        """Get total number of actions."""
        return self.action_size
    
    def get_dql_state_size(self, num_top_k_rsrp=3):
        """Get state size."""
        return self.state_size
    
    def get_dql_state_for_ue(
        self, ue: UserEquipment, bss_map, rb_pool: ResourceBlockPool, num_top_k_rsrp=3
    ):
        """Extract state vector for UE."""
        # Satisfaction state
        is_satisfied = ue.current_total_rate_mbps >= self.params.target_ue_throughput_mbps
        satisfaction_state = 1.0 if is_satisfied else 0.0
        
        # Load states for serving BSs
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
            
            if best_neigh_rsrp > rsrp_serv + self.params.ho_hysteresis_db + 6:
                rsrp_diff = 1.0
            elif best_neigh_rsrp > rsrp_serv + self.params.ho_hysteresis_db + 3:
                rsrp_diff = 0.75
            elif best_neigh_rsrp > rsrp_serv + self.params.ho_hysteresis_db:
                rsrp_diff = 0.5
            elif best_neigh_rsrp > rsrp_serv:
                rsrp_diff = 0.25
        
        # Current rate ratio
        rate_ratio = np.clip(
            ue.current_total_rate_mbps / (self.params.target_ue_throughput_mbps + 1e-6),
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
