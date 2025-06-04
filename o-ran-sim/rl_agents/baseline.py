import random
import numpy as np
from sim_core.params import SimParams
from sim_core.entities import UserEquipment, BaseStation
from sim_core.resource import ResourceBlockPool
from sim_core.helpers import linear_to_dbm, dbm_to_linear

from rl_agents.base import RLAgentBase

class BaselineAgent(RLAgentBase):
    def __init__(self, params: SimParams, num_bss, num_ues, logger_func, bss_map, rb_pool_ref):
        super().__init__(params, num_bss, num_ues, logger_func)
        self.bss_map_ref = bss_map
        self.rb_pool_ref = rb_pool_ref
        self.log("BaselineAgent initialized.")

    def get_ue_actions_and_allocate(self, ues_map):
        handovers_this_step = 0

        # 1. Handover Decisions (Simplified for single connectivity initially)
        for ue in ues_map.values():
            if ue.serving_bs_1 is None: # Initial association
                self._perform_initial_association(ue)
                continue

            current_serving_id = ue.serving_bs_1.id
            rsrp_serving = ue.measurements.get(current_serving_id, -np.inf)

            best_neighbor_id = None
            best_neighbor_rsrp = -np.inf

            # Find best non-serving candidate
            for bs_id, n_rsrp in ue.best_bs_candidates:
                if bs_id == current_serving_id:
                    continue
                if n_rsrp > best_neighbor_rsrp:
                    best_neighbor_rsrp = n_rsrp
                    best_neighbor_id = bs_id

            if best_neighbor_id:
                # A3 condition: Neighbor becomes offset better than serving
                a3_condition_met = (
                    best_neighbor_rsrp > rsrp_serving + self.params.ho_hysteresis_db
                )
                if a3_condition_met:
                    ue.time_in_a3_condition[best_neighbor_id] = (
                        ue.time_in_a3_condition.get(best_neighbor_id, 0) + 1
                    )
                else:
                    ue.time_in_a3_condition[best_neighbor_id] = 0

                # Time to Trigger condition
                ttt_met = (
                    ue.time_in_a3_condition.get(best_neighbor_id, 0)
                    >= self.params.ho_time_to_trigger_steps
                )
                # Acquisition threshold
                acq_met = best_neighbor_rsrp > self.params.min_rsrp_for_acq_dbm

                if a3_condition_met and ttt_met and acq_met:
                    old_bs = ue.serving_bs_1
                    new_bs = self.bss_map_ref.get(best_neighbor_id)
                    if new_bs:
                        self.log(
                            f"{ue.id} HANDOVER from {old_bs.id} to {new_bs.id} (RSRP_new: {best_neighbor_rsrp:.2f}, RSRP_old: {rsrp_serving:.2f})"
                        )
                        ue.serving_bs_1 = new_bs
                        ue.serving_bs_2 = None # Baseline is single connectivity
                        self.rb_pool_ref.release_rbs_for_ue(ue.id) # Release old RBs
                        ue.rbs_from_bs1.clear()
                        ue.rbs_from_bs2.clear()
                        ue.time_in_a3_condition.clear() # Reset TTT timers
                        handovers_this_step += 1

        # 2. Resource Allocation (Greedy, for serving_bs_1 only)
        # First, clear previous allocations from the pool for this step.
        # This is handled globally in Simulation.run_step, but each BS also tracks served UEs.
        for bs in self.bss_map_ref.values():
            bs.ues_served_this_step.clear()

        # Iterate UEs and allocate RBs
        for ue in ues_map.values():
            if ue.serving_bs_1:
                bs = ue.serving_bs_1
                bs.ues_served_this_step.add(ue.id)

                # Release all existing RBs for this UE from this BS for re-allocation
                # This ensures we don't carry over RBs that no longer make sense.
                # Already handled by ue.reset_connections_and_rates or _apply_rl_action_for_ue
                # For baseline, we just need to ensure UE's internal RB list is cleared before allocating.
                ue.rbs_from_bs1.clear() # Clear RBs *from this BS*
                ue.rbs_from_bs2.clear() # Ensure secondary is clear too

                # Greedy allocation: try to get max_rbs_per_ue_per_bs
                num_rbs_to_request = self.params.max_rbs_per_ue_per_bs
                
                # Check current potential rate vs target before requesting more RBs
                current_potential_rate_bps = 0
                temp_rbs = self.rb_pool_ref.get_available_rbs_for_bs(bs.id, num_rbs_to_request)
                
                for rb_id in temp_rbs:
                    sinr_l = bs.calculate_sinr_on_rb(ue, rb_id, self.bss_map_ref, self.rb_pool_ref)
                    current_potential_rate_bps += self.params.channel_model.shannon_capacity(sinr_l, self.params.rb_bandwidth_hz)

                # If current potential rate is already significantly above target, allocate fewer RBs
                if current_potential_rate_bps / 1e6 > self.params.target_ue_throughput_mbps * 1.2:
                    num_rbs_to_request = max(1, self.params.max_rbs_per_ue_per_bs // 2)
                
                available_rbs = self.rb_pool_ref.get_available_rbs_for_bs(bs.id, num_rbs_to_request)

                for rb_id in available_rbs:
                    if len(ue.rbs_from_bs1) >= self.params.max_rbs_per_ue_per_bs:
                        break # Reached max RBs from this BS
                    
                    # Allocate and mark
                    ue.rbs_from_bs1.append(rb_id)
                    self.rb_pool_ref.mark_allocated(rb_id, bs.id, ue.id)
        return handovers_this_step

    def _perform_initial_association(self, ue: UserEquipment):
        if not ue.measurements:
            return
        sorted_measurements = sorted(
            ue.measurements.items(), key=lambda item: item[1], reverse=True
        )
        for bs_id, rsrp in sorted_measurements:
            if rsrp > self.params.min_rsrp_for_acq_dbm:
                target_bs = self.bss_map_ref.get(bs_id)
                if target_bs:
                    ue.serving_bs_1 = target_bs
                    self.log(
                        f"{ue.id} initial association with {target_bs.id} (RSRP: {rsrp:.2f} dBm)"
                    )
                    return
        self.log(f"{ue.id} failed initial association.")