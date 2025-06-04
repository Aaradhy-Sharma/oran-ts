import random
import numpy as np
import math
from scipy.stats import poisson
from sim_core.params import SimParams
from sim_core.channel import ChannelModel
from sim_core.resource import ResourceBlockPool
from sim_core.entities import BaseStation, UserEquipment
# Import all RL Agents, dynamically loaded later
from rl_agents.base import RLAgentBase
from rl_agents.baseline import BaselineAgent
from rl_agents.tabular_q import TabularQLearningAgent
from rl_agents.sarsa import SARSAAgent
from rl_agents.expected_sarsa import ExpectedSARSAAgent
from rl_agents.nstep_sarsa import NStepSARSAAgent
from rl_agents.dqn import DQNAgent, TF_AVAILABLE_GLOBAL as TF_AVAILABLE # Get TF status

class Simulation:
    def __init__(self, params: SimParams, app_logger):
        self.params = params
        self.logger = app_logger
        self.channel_model = ChannelModel(params)
        self.rb_pool = ResourceBlockPool(params.num_total_rbs)
        self.params.channel_model = self.channel_model # Link channel model to params for UE access

        self.bss = {}
        self.ues = {}
        self.bs_id_to_idx_map = {}
        self.bs_idx_to_id_map = {}

        self._initialize_bs_ue_positions()

        # Initialize RL agent AFTER num_bss_actual and num_ues_actual are set
        self.rl_agent = self._initialize_rl_agent()

        self.current_time_step = 0
        self.total_handovers_cumulative = 0
        self.metrics_history = []

        self.last_ue_states_actions = {} # For SARSA variants to store S_t, A_t

    def _initialize_bs_ue_positions(self):
        area_m2 = self.params.sim_area_x * self.params.sim_area_y
        if self.params.placement_method == "PPP":
            num_bs_actual = poisson.rvs(self.params.lambda_bs * (area_m2 * 1e-6))
            if num_bs_actual == 0: num_bs_actual = 1
            self.log_message(f"PPP: Placing {num_bs_actual} BSs")
        else:
            num_bs_actual = self.params.num_bss
            self.log_message(f"Uniform: Placing {num_bs_actual} BSs")
        self.params.num_bss_actual = num_bs_actual

        for i in range(num_bs_actual):
            bs_id = f"BS{i}"
            pos_x = random.uniform(0.05 * self.params.sim_area_x, 0.95 * self.params.sim_area_x)
            pos_y = random.uniform(0.05 * self.params.sim_area_y, 0.95 * self.params.sim_area_y)
            bs = BaseStation(bs_id, (pos_x, pos_y), self.params, self.channel_model)
            self.bss[bs.id] = bs
            self.bs_id_to_idx_map[bs.id] = i
            self.bs_idx_to_id_map[i] = bs.id

        if self.params.placement_method == "PPP":
            num_ue_actual = poisson.rvs(self.params.lambda_ue * (area_m2 * 1e-6))
            if num_ue_actual == 0: num_ue_actual = 1
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
            return BaselineAgent(self.params, num_bss_act, num_ues_act, self.log_message, self.bss, self.rb_pool)
        elif agent_type == "TabularQLearning":
            return TabularQLearningAgent(self.params, num_bss_act, num_ues_act, self.log_message)
        elif agent_type == "SARSA":
            return SARSAAgent(self.params, num_bss_act, num_ues_act, self.log_message)
        elif agent_type == "ExpectedSARSA":
            return ExpectedSARSAAgent(self.params, num_bss_act, num_ues_act, self.log_message)
        elif agent_type == "NStepSARSA":
            n_step_val = getattr(self.params, "rl_n_step_sarsa", 3)
            return NStepSARSAAgent(self.params, num_bss_act, num_ues_act, self.log_message, n_step=n_step_val)
        elif agent_type == "DQL":
            if not TF_AVAILABLE:
                self.log_message("ERROR: TensorFlow not available for DQL. Falling back to Baseline.")
                self.params.rl_agent_type = "Baseline"
                return BaselineAgent(self.params, num_bss_act, num_ues_act, self.log_message, self.bss, self.rb_pool)
            temp_dql_for_sizes = DQNAgent(self.params, 10, 10, num_bss_act, self.log_message) # dummy init for size calcs
            state_size = temp_dql_for_sizes.get_dql_state_size()
            action_size = temp_dql_for_sizes.get_dql_action_space_size()
            del temp_dql_for_sizes
            return DQNAgent(self.params, state_size, action_size, num_bss_act, self.log_message)
        else:
            self.log_message(f"Unknown RL agent type: {agent_type}. Using Baseline.")
            self.params.rl_agent_type = "Baseline"
            return BaselineAgent(self.params, num_bss_act, num_ues_act, self.log_message, self.bss, self.rb_pool)

    def log_message(self, message):
        if self.logger:
            self.logger(message)
        else:
            print(message)

    def _apply_rl_action_for_ue(self, ue: UserEquipment, bs1_idx, bs2_idx, num_rbs_bs1, num_rbs_bs2):
        old_bs1_id = ue.serving_bs_1.id if ue.serving_bs_1 else None

        self.rb_pool.release_rbs_for_ue(ue.id)
        ue.rbs_from_bs1.clear()
        ue.rbs_from_bs2.clear()

        # Map indices back to BS objects
        ue.serving_bs_1 = (
            self.bss.get(self.bs_idx_to_id_map.get(bs1_idx)) if bs1_idx != -1 else None
        )
        ue.serving_bs_2 = (
            self.bss.get(self.bs_idx_to_id_map.get(bs2_idx)) if bs2_idx != -1 else None
        )

        # Ensure BS2 is different from BS1 if both are active and assigned
        if ue.serving_bs_1 and ue.serving_bs_2 and ue.serving_bs_1.id == ue.serving_bs_2.id:
            ue.serving_bs_2 = None
            num_rbs_bs2 = 0

        # Adjust RB counts if BS is None
        if ue.serving_bs_1 is None:
            num_rbs_bs1 = 0
        if ue.serving_bs_2 is None:
            num_rbs_bs2 = 0

        # Cap total RBs if combined exceeds max_rbs_per_ue
        if num_rbs_bs1 + num_rbs_bs2 > self.params.max_rbs_per_ue:
            if num_rbs_bs1 >= self.params.max_rbs_per_ue: # Prioritize BS1 if over limit
                num_rbs_bs1 = self.params.max_rbs_per_ue
                num_rbs_bs2 = 0
            else: # Reduce BS2 RBs to fit
                num_rbs_bs2 = self.params.max_rbs_per_ue - num_rbs_bs1

        # Allocate RBs for BS1
        if ue.serving_bs_1 and num_rbs_bs1 > 0:
            available_rbs = self.rb_pool.get_available_rbs_for_bs(
                ue.serving_bs_1.id, num_rbs_bs1
            )
            for rb_id in available_rbs:
                ue.rbs_from_bs1.append(rb_id)
                self.rb_pool.mark_allocated(rb_id, ue.serving_bs_1.id, ue.id)
            if len(available_rbs) < num_rbs_bs1:
                self.log_message(
                    f"Warning: {ue.id} wanted {num_rbs_bs1} from {ue.serving_bs_1.id}, got {len(available_rbs)}"
                )
            ue.serving_bs_1.ues_served_this_step.add(ue.id)

        # Allocate RBs for BS2
        if ue.serving_bs_2 and num_rbs_bs2 > 0:
            available_rbs = self.rb_pool.get_available_rbs_for_bs(
                ue.serving_bs_2.id, num_rbs_bs2
            )
            for rb_id in available_rbs:
                ue.rbs_from_bs2.append(rb_id)
                self.rb_pool.mark_allocated(rb_id, ue.serving_bs_2.id, ue.id)
            if len(available_rbs) < num_rbs_bs2:
                self.log_message(
                    f"Warning: {ue.id} wanted {num_rbs_bs2} from {ue.serving_bs_2.id}, got {len(available_rbs)}"
                )
            ue.serving_bs_2.ues_served_this_step.add(ue.id)

        # Check for handover (change in primary BS)
        new_bs1_id = ue.serving_bs_1.id if ue.serving_bs_1 else None
        if (
            old_bs1_id != new_bs1_id
            and old_bs1_id is not None
            and new_bs1_id is not None
        ):
            return 1
        return 0

    def _calculate_reward(self, handovers_this_step):
        total_ue_satisfaction_score = 0.0
        num_connected_ues = 0

        for ue in self.ues.values():
            if ue.serving_bs_1 or ue.serving_bs_2:
                num_connected_ues += 1
                if ue.current_total_rate_mbps >= self.params.target_ue_throughput_mbps:
                    total_ue_satisfaction_score += 1.0
                elif ue.current_total_rate_mbps > 0.5 * self.params.target_ue_throughput_mbps:
                    total_ue_satisfaction_score += 0.5
                elif ue.current_total_rate_mbps > 0:
                    total_ue_satisfaction_score += 0.1

        avg_ue_satisfaction_reward = (
            total_ue_satisfaction_score / self.params.num_ues_actual
            if self.params.num_ues_actual > 0
            else 0.0
        )

        total_bs_load_penalty = 0.0
        num_active_bss = 0
        for bs in self.bss.values():
            if bs.load_factor_metric > 0:
                num_active_bss += 1
                total_bs_load_penalty += (bs.load_factor_metric - 1.0) ** 2

        avg_bs_load_penalty = (
            total_bs_load_penalty / num_active_bss if num_active_bss > 0 else 0.0
        )
        reward_bs_load = 1.0 - avg_bs_load_penalty

        ho_penalty = -0.1 * handovers_this_step
        ho_penalty = max(ho_penalty, -1.0) # Cap penalty

        w_ue_satisfaction = 0.6
        w_bs_load = 0.3
        w_ho_penalty = 0.1

        total_reward = (
            w_ue_satisfaction * avg_ue_satisfaction_reward
            + w_bs_load * reward_bs_load
            + w_ho_penalty * ho_penalty
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
            current_handovers_this_step = self.rl_agent.get_ue_actions_and_allocate(self.ues)
        else:
            # Store S_t, A_t for each UE for later training
            # This is specifically for SARSA-like updates where A_t+1 is needed.
            # For Q-learning, this `last_ue_states_actions` isn't strictly necessary for the update,
            # but it's good to keep a consistent experience structure.
            new_last_ue_states_actions = {}
            for ue_id, ue_obj in self.ues.items():
                s_ue = None
                a_idx_ue = -1

                if self.params.rl_agent_type in ["TabularQLearning", "SARSA", "ExpectedSARSA", "NStepSARSA"]:
                    s_ue = self.rl_agent._get_state_representation(ue_obj, self.bss)
                    a_idx_ue = self.rl_agent.get_action(s_ue, ue_id)
                elif self.params.rl_agent_type == "DQL":
                    s_ue = self.rl_agent.get_dql_state_for_ue(ue_obj, self.bss, self.rb_pool)
                    a_idx_ue = self.rl_agent.get_action(s_ue, ue_id)

                new_last_ue_states_actions[ue_id] = {"s": s_ue, "a": a_idx_ue}

                # Apply chosen actions (A_t)
                bs1_act_idx, bs2_act_idx, rbs1_act, rbs2_act = -1, -1, 0, 0
                if self.params.rl_agent_type in ["TabularQLearning", "SARSA", "ExpectedSARSA", "NStepSARSA"]:
                    current_bs1_idx = (
                        self.bs_id_to_idx_map.get(ue_obj.serving_bs_1.id)
                        if ue_obj.serving_bs_1
                        else -1
                    )
                    if a_idx_ue == 0:  # Stay
                        bs1_act_idx = current_bs1_idx
                        bs2_act_idx = (
                            self.bs_id_to_idx_map.get(ue_obj.serving_bs_2.id)
                            if ue_obj.serving_bs_2
                            else -1
                        )
                        rbs1_act = len(ue_obj.rbs_from_bs1)
                        rbs2_act = len(ue_obj.rbs_from_bs2)
                    elif a_idx_ue == 1:  # Switch primary to best candidate
                        best_cand_id = (
                            ue_obj.best_bs_candidates[0][0]
                            if ue_obj.best_bs_candidates
                            else None
                        )
                        bs1_act_idx = (
                            self.bs_id_to_idx_map.get(best_cand_id)
                            if best_cand_id
                            else -1
                        )
                        bs2_act_idx = -1 # Single connectivity
                        rbs1_act = self.params.max_rbs_per_ue_per_bs
                        rbs2_act = 0
                    elif a_idx_ue == 2:  # Try dual connectivity with best candidate
                        bs1_act_idx = current_bs1_idx
                        best_cand_id_2nd = None
                        if ue_obj.best_bs_candidates:
                            # Find best non-serving candidate for secondary link
                            for cand_id, _ in ue_obj.best_bs_candidates:
                                if not ue_obj.serving_bs_1 or cand_id != ue_obj.serving_bs_1.id:
                                    best_cand_id_2nd = cand_id
                                    break
                        bs2_act_idx = (
                            self.bs_id_to_idx_map.get(best_cand_id_2nd)
                            if best_cand_id_2nd
                            else -1
                        )

                        # Adjust RB distribution for dual connectivity
                        total_allowed_rbs = self.params.max_rbs_per_ue
                        if bs1_act_idx != -1 and bs2_act_idx != -1 and bs1_act_idx != bs2_act_idx:
                            # Divide available RBs, but don't exceed max_rbs_per_ue_per_bs per BS
                            rbs1_act = min(self.params.max_rbs_per_ue_per_bs, math.ceil(total_allowed_rbs / 2))
                            rbs2_act = min(self.params.max_rbs_per_ue_per_bs, total_allowed_rbs - rbs1_act)
                        elif bs1_act_idx != -1: # Only BS1, single connectivity
                            rbs1_act = self.params.max_rbs_per_ue
                            rbs2_act = 0
                        elif bs2_act_idx != -1: # Only BS2 (as primary if no BS1), single connectivity
                            bs1_act_idx = bs2_act_idx # Make BS2 the primary if BS1 is None
                            rbs1_act = self.params.max_rbs_per_ue
                            bs2_act_idx = -1
                            rbs2_act = 0
                        else: # No connection
                            rbs1_act = 0
                            rbs2_act = 0

                elif self.params.rl_agent_type == "DQL":
                    (
                        bs1_act_idx,
                        bs2_act_idx,
                        rbs1_act,
                        rbs2_act,
                    ) = self.rl_agent.map_action_idx_to_ue_config(a_idx_ue)
                
                ho_event = self._apply_rl_action_for_ue(
                    ue_obj, bs1_act_idx, bs2_act_idx, rbs1_act, rbs2_act
                )
                current_handovers_this_step += ho_event

            self.last_ue_states_actions = new_last_ue_states_actions # Update for next step's training

        # Update UE Rates/SINRs, BS Loads (R_t+1 is observed globally)
        for ue in self.ues.values():
            ue.update_ue_rates_sinrs(self.bss, self.rb_pool)
        for bs in self.bss.values():
            bs.update_load_factor(self.ues)

        current_reward = self._calculate_reward(current_handovers_this_step)
        self.log_message(f"Step Reward: {current_reward:.3f}")

        # Train agents (S_t+1 and A_t+1 for SARSA-like updates)
        if self.params.rl_agent_type != "Baseline":
            self.rl_agent.update_epsilon()
            for ue_id, ue_obj in self.ues.items():
                experience = self.last_ue_states_actions.get(ue_id)
                if not experience: # Should not happen after first step
                    continue

                s_t = experience["s"]
                a_t = experience["a"]

                s_prime_t = None
                a_prime_t = -1 # For SARSA, NStepSARSA

                if self.params.rl_agent_type in ["TabularQLearning", "SARSA", "ExpectedSARSA", "NStepSARSA"]:
                    s_prime_t = self.rl_agent._get_state_representation(ue_obj, self.bss)
                    if self.params.rl_agent_type in ["SARSA", "NStepSARSA"]:
                        # Choose A_t+1 based on S_t+1 according to current policy for SARSA
                        a_prime_t = self.rl_agent.get_action(s_prime_t, ue_id) # This call also updates NStep's next_action for trajectory
                elif self.params.rl_agent_type == "DQL":
                    s_prime_t = self.rl_agent.get_dql_state_for_ue(ue_obj, self.bss, self.rb_pool)


                if self.params.rl_agent_type == "TabularQLearning":
                    self.rl_agent.train(s_t, a_t, current_reward, s_prime_t, False, ue_id)
                elif self.params.rl_agent_type == "SARSA":
                    self.rl_agent.train(s_t, a_t, current_reward, s_prime_t, a_prime_t, False, ue_id)
                elif self.params.rl_agent_type == "ExpectedSARSA":
                    self.rl_agent.train(s_t, a_t, current_reward, s_prime_t, False, ue_id)
                elif self.params.rl_agent_type == "NStepSARSA":
                    self.rl_agent.record_experience_and_train(ue_id, s_t, a_t, current_reward, False)
                elif self.params.rl_agent_type == "DQL":
                    self.rl_agent.remember(s_t, a_t, current_reward, s_prime_t, False)

            if self.params.rl_agent_type == "DQL":
                self.rl_agent.train()
                if self.current_time_step % self.params.rl_target_update_freq == 0:
                    self.rl_agent.update_target_model()

        self.total_handovers_cumulative += current_handovers_this_step
        step_metrics = self.calculate_step_metrics(current_handovers_this_step, current_reward)
        self.metrics_history.append(step_metrics)
        self.log_step_metrics(step_metrics)

        self.current_time_step += 1
        return self.current_time_step < self.params.total_sim_steps

    def calculate_step_metrics(self, handovers_this_step, current_reward=0.0):
        actual_num_ues_for_calc = self.params.num_ues_actual

        connected_ue_rates = [
            ue.current_total_rate_mbps
            for ue in self.ues.values()
            if (ue.serving_bs_1 or ue.serving_bs_2) and ue.current_total_rate_mbps > 0
        ]

        connected_ue_sinrs_db = []
        for ue in self.ues.values():
            if ue.serving_bs_1 and ue.sinr_db_bs1 > -np.inf:
                connected_ue_sinrs_db.append(ue.sinr_db_bs1)
            elif ue.serving_bs_2 and ue.sinr_db_bs2 > -np.inf:
                connected_ue_sinrs_db.append(ue.sinr_db_bs2)

        connected_ue_rbs_counts = [
            len(ue.rbs_from_bs1) + len(ue.rbs_from_bs2)
            for ue in self.ues.values()
            if (ue.serving_bs_1 or ue.serving_bs_2)
        ]

        connected_ue_rsrps = []
        if self.ues and self.bss:
            for ue in self.ues.values():
                if ue.serving_bs_1 and ue.measurements:
                    val = ue.measurements.get(ue.serving_bs_1.id, -np.inf)
                    if val > -np.inf: connected_ue_rsrps.append(val)

        total_throughput_mbps = sum(connected_ue_rates)
        num_actually_connected_ues = sum(
            1 for ue in self.ues.values() if (ue.serving_bs_1 or ue.serving_bs_2)
        )
        num_satisfied_ues = sum(
            1
            for ue in self.ues.values()
            if (ue.serving_bs_1 or ue.serving_bs_2)
            and ue.current_total_rate_mbps >= self.params.target_ue_throughput_mbps
        )

        avg_bs_load_factor = (
            np.mean([bs.load_factor_metric for bs in self.bss.values()])
            if self.bss
            else 0.0
        )

        ue_details = [] # Optional: for detailed logging of each UE if needed
        # for ue in self.ues.values():
        #     s_bs1 = ue.serving_bs_1.id if ue.serving_bs_1 else "N"
        #     s_bs2 = ue.serving_bs_2.id if ue.serving_bs_2 else "N"
        #     ue_d = {
        #         "id": ue.id,
        #         "pos": (round(ue.position[0], 1), round(ue.position[1], 1)),
        #         "serving_bs": f"{s_bs1}/{s_bs2}",
        #         "rate_mbps": round(ue.current_total_rate_mbps, 2),
        #         "sinr_db": round(
        #             ue.sinr_db_bs1 if ue.serving_bs_1 else ue.sinr_db_bs2, 2
        #         ),
        #         "rbs": len(ue.rbs_from_bs1) + len(ue.rbs_from_bs2),
        #         "best_cand": ue.best_bs_candidates[:2],
        #     }
        #     ue_details.append(ue_d)

        return {
            "time_step": self.current_time_step,
            "reward": current_reward,
            "total_throughput_mbps": total_throughput_mbps,
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
            "avg_bs_load_factor": avg_bs_load_factor,
            "ue_details": ue_details, # Keep empty if not needed to save memory
            "num_connected_ues": num_actually_connected_ues,
            "sinr_min_db": np.min(connected_ue_sinrs_db)
            if connected_ue_sinrs_db
            else -np.inf,
            "sinr_max_db": np.max(connected_ue_sinrs_db)
            if connected_ue_sinrs_db
            else -np.inf,
            "sinr_avg_db": np.mean(connected_ue_sinrs_db)
            if connected_ue_sinrs_db
            else -np.inf,
            "rbs_avg_per_ue": np.mean(connected_ue_rbs_counts)
            if connected_ue_rbs_counts
            else 0.0,
            "rsrp_avg_dbm": np.mean(connected_ue_rsrps)
            if connected_ue_rsrps
            else -np.inf,
            "actual_num_ues": actual_num_ues_for_calc,
            "epsilon": self.rl_agent.current_epsilon
            if hasattr(self.rl_agent, "current_epsilon")
            else -1.0,
        }

    def log_step_metrics(self, metrics):
        self.log_message(
            f"Metrics @ T{metrics['time_step']}: Rew:{metrics['reward']:.2f} Thr:{metrics['total_throughput_mbps']:.2f} "
            f"Satisfied:{metrics['num_satisfied_ues']}({metrics['percentage_satisfied_ues']:.1f}%) "
            f"HOs:{metrics['handovers_this_step']} Load:{metrics['avg_bs_load_factor']:.2f} "
            f"AvgSINR:{metrics['sinr_avg_db']:.2f} Eps:{metrics['epsilon']:.3f}"
        )