import random
import numpy as np
import math
from scipy.stats import poisson

from sim_core.params import SimParams
from sim_core.channel import ChannelModel
from sim_core.resource import ResourceBlockPool
from sim_core.entities import BaseStation, UserEquipment
from rl_agents.base import RLAgentBase
from rl_agents.baseline import BaselineAgent
from rl_agents.tabular_q import TabularQLearningAgent
from rl_agents.sarsa import SARSAAgent
from rl_agents.expected_sarsa import ExpectedSARSAAgent
from rl_agents.nstep_sarsa import NStepSARSAAgent

try:
    from rl_agents.dqn import DQNAgent
    TF_AVAILABLE = True
except ImportError:
    DQNAgent = None
    TF_AVAILABLE = False


class Simulation:
    def __init__(self, params: SimParams, app_logger):
        self.params = params
        self.logger = app_logger
        self.channel_model = ChannelModel(params)
        self.rb_pool = ResourceBlockPool(params.num_total_rbs)
        self.params.channel_model = self.channel_model

        self.bss = {}
        self.ues = {}
        self.bs_id_to_idx_map = {}
        self.bs_idx_to_id_map = {}

        self._initialize_bs_ue_positions()
        self.rl_agent = self._initialize_rl_agent()

        self.current_time_step = 0
        self.total_handovers_cumulative = 0
        self.metrics_history = []
        self.last_ue_states_actions = {}

    def _initialize_bs_ue_positions(self):
        area_m2 = self.params.sim_area_x * self.params.sim_area_y
        if self.params.placement_method == "PPP":
            num_bs_actual = poisson.rvs(self.params.lambda_bs * (area_m2 * 1e-6))
            if num_bs_actual == 0:
                num_bs_actual = 1
            self.log_message(f"PPP: Placing {num_bs_actual} BSs")
        else:
            num_bs_actual = self.params.num_bss
            self.log_message(f"Uniform: Placing {num_bs_actual} BSs")
        self.params.num_bss_actual = num_bs_actual

        for i in range(num_bs_actual):
            bs_id = f"BS{i}"
            pos_x = random.uniform(
                0.05 * self.params.sim_area_x, 0.95 * self.params.sim_area_x
            )
            pos_y = random.uniform(
                0.05 * self.params.sim_area_y, 0.95 * self.params.sim_area_y
            )
            bs = BaseStation(bs_id, (pos_x, pos_y), self.params, self.channel_model)
            self.bss[bs.id] = bs
            self.bs_id_to_idx_map[bs.id] = i
            self.bs_idx_to_id_map[i] = bs.id

        if self.params.placement_method == "PPP":
            num_ue_actual = poisson.rvs(self.params.lambda_ue * (area_m2 * 1e-6))
            if num_ue_actual == 0:
                num_ue_actual = 1
            self.log_message(f"PPP: Placing {num_ue_actual} UEs")
        else:
            num_ue_actual = self.params.num_ues
            self.log_message(f"Uniform: Placing {num_ue_actual} UEs")
        self.params.num_ues_actual = num_ue_actual

        for i in range(num_ue_actual):
            pos = (
                random.uniform(0, self.params.sim_area_x),
                random.uniform(0, self.params.sim_area_y),
            )
            ue = UserEquipment(f"UE{i}", pos, self.params)
            self.ues[ue.id] = ue

    def _initialize_rl_agent(self):
        agent_type = self.params.rl_agent_type
        num_bss_act = self.params.num_bss_actual
        num_ues_act = self.params.num_ues_actual

        if agent_type == "Baseline":
            return BaselineAgent(
                self.params, num_bss_act, num_ues_act, self.log_message, self.bss, self.rb_pool
            )
        elif agent_type == "TabularQLearning":
            return TabularQLearningAgent(
                self.params, num_bss_act, num_ues_act, self.log_message
            )
        elif agent_type == "SARSA":
            return SARSAAgent(self.params, num_bss_act, num_ues_act, self.log_message)
        elif agent_type == "ExpectedSARSA":
            return ExpectedSARSAAgent(
                self.params, num_bss_act, num_ues_act, self.log_message
            )
        elif agent_type == "NStepSARSA":
            n_step_val = getattr(self.params, "rl_n_step_sarsa", 3)
            return NStepSARSAAgent(
                self.params, num_bss_act, num_ues_act, self.log_message, n_step=n_step_val
            )
        elif agent_type == "DQN":
            if not TF_AVAILABLE:
                raise ImportError(
                    "TensorFlow is required for DQN agent but is not installed."
                )
            temp_dql = DQNAgent(self.params, 10, 10, num_bss_act, self.log_message)
            state_size = temp_dql.get_dql_state_size()
            action_size = temp_dql.get_dql_action_space_size()
            del temp_dql
            return DQNAgent(
                self.params, state_size, action_size, num_bss_act, self.log_message
            )
        else:
            self.log_message(f"Unknown RL agent type: {agent_type}. Using Baseline.")
            self.params.rl_agent_type = "Baseline"
            return BaselineAgent(
                self.params, num_bss_act, num_ues_act, self.log_message, self.bss, self.rb_pool
            )

    def log_message(self, message):
        if self.logger:
            self.logger(message)
        else:
            print(message)

    def _map_tabular_action_to_config(self, ue, action_idx):
        current_bs1_idx = (
            self.bs_id_to_idx_map.get(ue.serving_bs_1.id) if ue.serving_bs_1 else -1
        )
        current_bs2_idx = (
            self.bs_id_to_idx_map.get(ue.serving_bs_2.id) if ue.serving_bs_2 else -1
        )

        if action_idx == 0:  # Stay
            return (
                current_bs1_idx,
                current_bs2_idx,
                len(ue.rbs_from_bs1),
                len(ue.rbs_from_bs2),
            )
        elif action_idx == 1:  # Switch primary to best candidate
            best_cand_id = ue.best_bs_candidates[0][0] if ue.best_bs_candidates else None
            bs1_idx = self.bs_id_to_idx_map.get(best_cand_id) if best_cand_id else -1
            return bs1_idx, -1, self.params.max_rbs_per_ue, 0
        elif action_idx == 2:  # Try dual connectivity
            bs1_idx = current_bs1_idx
            best_non_serving_cand_id = None
            if ue.best_bs_candidates:
                for cand_id, _ in ue.best_bs_candidates:
                    if not ue.serving_bs_1 or cand_id != ue.serving_bs_1.id:
                        best_non_serving_cand_id = cand_id
                        break
            bs2_idx = (
                self.bs_id_to_idx_map.get(best_non_serving_cand_id)
                if best_non_serving_cand_id
                else -1
            )

            rbs1, rbs2 = 0, 0
            if bs1_idx != -1 and bs2_idx != -1:  # Dual connect
                rbs1 = min(
                    self.params.max_rbs_per_ue_per_bs,
                    math.ceil(self.params.max_rbs_per_ue / 2),
                )
                rbs2 = min(
                    self.params.max_rbs_per_ue_per_bs, self.params.max_rbs_per_ue - rbs1
                )
            elif bs1_idx != -1:  # Only primary
                rbs1 = self.params.max_rbs_per_ue
            return bs1_idx, bs2_idx, rbs1, rbs2
        return -1, -1, 0, 0

    def _apply_rl_action_for_ue(
        self, ue: UserEquipment, bs1_idx, bs2_idx, num_rbs_bs1, num_rbs_bs2
    ):
        old_bs1_id = ue.serving_bs_1.id if ue.serving_bs_1 else None

        self.rb_pool.release_rbs_for_ue(ue.id)
        ue.rbs_from_bs1.clear()
        ue.rbs_from_bs2.clear()

        ue.serving_bs_1 = (
            self.bss.get(self.bs_idx_to_id_map.get(bs1_idx)) if bs1_idx != -1 else None
        )
        ue.serving_bs_2 = (
            self.bss.get(self.bs_idx_to_id_map.get(bs2_idx)) if bs2_idx != -1 else None
        )

        if ue.serving_bs_1 and ue.serving_bs_2 and ue.serving_bs_1.id == ue.serving_bs_2.id:
            ue.serving_bs_2 = None
            num_rbs_bs2 = 0

        if ue.serving_bs_1 is None:
            num_rbs_bs1 = 0
        if ue.serving_bs_2 is None:
            num_rbs_bs2 = 0

        if ue.serving_bs_1 and num_rbs_bs1 > 0:
            available_rbs = self.rb_pool.get_available_rbs_for_bs(
                ue.serving_bs_1.id, num_rbs_bs1
            )
            for rb_id in available_rbs:
                ue.rbs_from_bs1.append(rb_id)
                self.rb_pool.mark_allocated(rb_id, ue.serving_bs_1.id, ue.id)
            ue.serving_bs_1.ues_served_this_step.add(ue.id)

        if ue.serving_bs_2 and num_rbs_bs2 > 0:
            available_rbs = self.rb_pool.get_available_rbs_for_bs(
                ue.serving_bs_2.id, num_rbs_bs2
            )
            for rb_id in available_rbs:
                ue.rbs_from_bs2.append(rb_id)
                self.rb_pool.mark_allocated(rb_id, ue.serving_bs_2.id, ue.id)
            ue.serving_bs_2.ues_served_this_step.add(ue.id)

        new_bs1_id = ue.serving_bs_1.id if ue.serving_bs_1 else None
        if old_bs1_id != new_bs1_id and old_bs1_id is not None and new_bs1_id is not None:
            return 1
        return 0

    def _calculate_reward(self, handovers_this_step):
        # Calculate UE satisfaction score (binary satisfaction)
        total_ue_satisfaction_score = 0.0
        total_throughput = 0.0
        max_possible_throughput = self.params.target_ue_throughput_mbps * self.params.num_ues_actual
        
        for ue in self.ues.values():
            if ue.serving_bs_1 or ue.serving_bs_2:
                # Add to total throughput
                total_throughput += ue.current_total_rate_mbps
                
                # Calculate satisfaction score
                if ue.current_total_rate_mbps >= self.params.target_ue_throughput_mbps:
                    total_ue_satisfaction_score += 1.0
                elif ue.current_total_rate_mbps > 0.5 * self.params.target_ue_throughput_mbps:
                    total_ue_satisfaction_score += 0.5
                elif ue.current_total_rate_mbps > 0:
                    total_ue_satisfaction_score += 0.1

        # Normalize satisfaction score
        avg_ue_satisfaction_reward = (
            total_ue_satisfaction_score / self.params.num_ues_actual
            if self.params.num_ues_actual > 0
            else 0.0
        )
        
        # Calculate throughput reward (normalized to [0,1])
        throughput_reward = total_throughput / max_possible_throughput if max_possible_throughput > 0 else 0.0
        
        # Calculate load penalties
        load_penalties = [
            (bs.load_factor_metric - 1.0) ** 2
            for bs in self.bss.values()
            if bs.load_factor_metric > 0
        ]
        avg_bs_load_penalty = np.mean(load_penalties) if load_penalties else 0.0
        reward_bs_load = 1.0 - avg_bs_load_penalty

        # Calculate handover penalty
        ho_penalty = -0.1 * handovers_this_step
        ho_penalty = max(ho_penalty, -1.0)

        # Updated weights to emphasize throughput
        w_throughput = 0.4  # New weight for throughput
        w_ue_satisfaction = 0.3  # Reduced from 0.6
        w_bs_load = 0.2  # Reduced from 0.3
        w_ho_penalty = 0.1  # Kept the same

        # Calculate total reward with throughput component
        total_reward = (
            w_throughput * throughput_reward +
            w_ue_satisfaction * avg_ue_satisfaction_reward +
            w_bs_load * reward_bs_load +
            w_ho_penalty * ho_penalty
        )
        
        return total_reward

    def run_step(self):
        self.log_message(f"\n--- Time Step {self.current_time_step} ---")
        self.rb_pool.reset()
        for bs in self.bss.values():
            bs.ues_served_this_step.clear()

        current_handovers_this_step = 0

        for ue in self.ues.values():
            ue.move()
            ue.perform_measurements(self.bss)

        if self.params.rl_agent_type == "Baseline":
            current_handovers_this_step = self.rl_agent.get_ue_actions_and_allocate(
                self.ues
            )
        else:
            new_last_ue_states_actions = {}
            for ue_id, ue_obj in self.ues.items():
                if self.params.rl_agent_type == "DQN":
                    s_ue = self.rl_agent.get_dql_state_for_ue(
                        ue_obj, self.bss, self.rb_pool
                    )
                    a_idx = self.rl_agent.get_action(s_ue, ue_id)
                    config = self.rl_agent.map_action_idx_to_ue_config(a_idx)
                else:  # Tabular agents
                    s_ue = self.rl_agent._get_state_representation(ue_obj, self.bss)
                    a_idx = self.rl_agent.get_action(s_ue, ue_id)
                    config = self._map_tabular_action_to_config(ue_obj, a_idx)

                new_last_ue_states_actions[ue_id] = {"s": s_ue, "a": a_idx}
                ho_event = self._apply_rl_action_for_ue(ue_obj, *config)
                current_handovers_this_step += ho_event

            self.last_ue_states_actions = new_last_ue_states_actions

        for ue in self.ues.values():
            ue.update_ue_rates_sinrs(self.bss, self.rb_pool)
        for bs in self.bss.values():
            bs.update_load_factor(self.ues)

        current_reward = self._calculate_reward(current_handovers_this_step)
        self.log_message(f"Step Reward: {current_reward:.3f}")

        # --- REFACTORED TRAINING LOGIC ---
        if self.params.rl_agent_type != "Baseline":
            self.rl_agent.update_epsilon()
            for ue_id, ue_obj in self.ues.items():
                experience = self.last_ue_states_actions.get(ue_id)
                if not experience:
                    continue

                s_t, a_t = experience["s"], experience["a"]

                # Group state calculation and training call by agent type
                if self.params.rl_agent_type == "DQN":
                    s_prime_t = self.rl_agent.get_dql_state_for_ue(
                        ue_obj, self.bss, self.rb_pool
                    )
                    self.rl_agent.remember(s_t, a_t, current_reward, s_prime_t, False)
                else:
                    # All other agents are tabular and use the same state representation
                    s_prime_t = self.rl_agent._get_state_representation(ue_obj, self.bss)

                    if self.params.rl_agent_type == "TabularQLearning":
                        self.rl_agent.train(s_t, a_t, current_reward, s_prime_t, False, ue_id)
                    elif self.params.rl_agent_type == "SARSA":
                        a_prime_t = self.rl_agent.get_action(s_prime_t, ue_id)
                        self.rl_agent.train(s_t, a_t, current_reward, s_prime_t, a_prime_t, False, ue_id)
                    elif self.params.rl_agent_type == "ExpectedSARSA":
                        self.rl_agent.train(s_t, a_t, current_reward, s_prime_t, False, ue_id)
                    elif self.params.rl_agent_type == "NStepSARSA":
                        # NStepSARSA needs the next action for its internal logic
                        self.rl_agent.get_action(s_prime_t, ue_id)
                        self.rl_agent.record_experience_and_train(ue_id, s_t, a_t, current_reward, False)

            # DQL batch training is done once per step, after collecting all experiences
            if self.params.rl_agent_type == "DQN":
                self.rl_agent.train()
                if self.current_time_step % self.params.rl_target_update_freq == 0:
                    self.rl_agent.update_target_model()

        self.total_handovers_cumulative += current_handovers_this_step
        step_metrics = self.calculate_step_metrics(
            current_handovers_this_step, current_reward
        )
        self.metrics_history.append(step_metrics)
        self.log_step_metrics(step_metrics)

        self.current_time_step += 1
        return self.current_time_step < self.params.total_sim_steps

    def calculate_step_metrics(self, handovers_this_step, current_reward=0.0):
        connected_ue_rates = [
            ue.current_total_rate_mbps
            for ue in self.ues.values()
            if (ue.serving_bs_1 or ue.serving_bs_2) and ue.current_total_rate_mbps > 0
        ]
        num_actually_connected_ues = sum(
            1 for ue in self.ues.values() if (ue.serving_bs_1 or ue.serving_bs_2)
        )
        num_satisfied_ues = sum(
            1
            for ue in self.ues.values()
            if (ue.serving_bs_1 or ue.serving_bs_2)
            and ue.current_total_rate_mbps >= self.params.target_ue_throughput_mbps
        )

        # Calculate SINR metrics
        sinr_values = []
        for ue in self.ues.values():
            if ue.serving_bs_1 or ue.serving_bs_2:
                if ue.serving_bs_1:
                    sinr_values.append(ue.measurements.get(ue.serving_bs_1.id, -np.inf))
                if ue.serving_bs_2:
                    sinr_values.append(ue.measurements.get(ue.serving_bs_2.id, -np.inf))
        
        # Calculate RBs per UE
        rbs_per_ue = []
        for ue in self.ues.values():
            if ue.serving_bs_1 or ue.serving_bs_2:
                total_rbs = len(ue.rbs_from_bs1) + len(ue.rbs_from_bs2)
                if total_rbs > 0:
                    rbs_per_ue.append(total_rbs)

        return {
            "time_step": self.current_time_step,
            "reward": current_reward,
            "total_throughput_mbps": sum(connected_ue_rates),
            "avg_ue_throughput_mbps": np.mean(connected_ue_rates)
            if connected_ue_rates
            else 0.0,
            "num_satisfied_ues": num_satisfied_ues,
            "percentage_satisfied_ues": (
                num_satisfied_ues / num_actually_connected_ues * 100
            )
            if num_actually_connected_ues > 0
            else 0.0,
            "handovers_this_step": handovers_this_step,
            "total_handovers_cumulative": self.total_handovers_cumulative,
            "avg_bs_load_factor": np.mean(
                [bs.load_factor_metric for bs in self.bss.values()]
            )
            if self.bss
            else 0.0,
            "num_connected_ues": num_actually_connected_ues,
            "epsilon": self.rl_agent.current_epsilon
            if hasattr(self.rl_agent, "current_epsilon")
            else -1.0,
            "sinr_avg_db": np.mean([s for s in sinr_values if s != -np.inf])
            if sinr_values
            else 0.0,
            "rbs_avg_per_ue": np.mean(rbs_per_ue)
            if rbs_per_ue
            else 0.0,
        }

    def log_step_metrics(self, metrics):
        self.log_message(
            f"Metrics @ T{metrics['time_step']}: Rew:{metrics['reward']:.2f} Thr:{metrics['total_throughput_mbps']:.2f} "
            f"Satisfied:{metrics['num_satisfied_ues']}({metrics['percentage_satisfied_ues']:.1f}%) "
            f"HOs:{metrics['handovers_this_step']} Load:{metrics['avg_bs_load_factor']:.2f} "
            f"Eps:{metrics['epsilon']:.3f}"
        )