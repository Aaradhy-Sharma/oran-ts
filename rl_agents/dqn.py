import collections
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sim_core.params import SimParams
from rl_agents.rla_base import RLAgentBase

class DQNAgent(RLAgentBase):
    def __init__(self, params: SimParams, state_size, action_size, num_bss_total, logger_func):
        super().__init__(params, num_bss_total, params.num_ues_actual, logger_func)
        self.state_size = state_size
        self.action_size = action_size
        self.num_bss_total = num_bss_total
        self.num_rb_categories = 3

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
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.params.rl_hidden_units, activation="relu", input_shape=(self.state_size,)))
        model.add(BatchNormalization())
        model.add(Dropout(self.params.rl_dropout_rate))
        
        # Hidden layers
        for _ in range(self.params.rl_num_hidden_layers - 1):
            model.add(Dense(self.params.rl_hidden_units, activation="relu"))
            model.add(BatchNormalization())
            model.add(Dropout(self.params.rl_dropout_rate))
        
        # Output layer
        model.add(Dense(self.action_size, activation="linear"))
        
        model.compile(
            optimizer=Adam(learning_rate=self.params.rl_learning_rate),
            loss="mse"
        )
        return model

    # ... rest of the existing methods ... 