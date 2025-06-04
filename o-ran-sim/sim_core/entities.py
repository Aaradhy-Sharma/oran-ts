import numpy as np
import math
from scipy.stats import rayleigh # NEW: For small-scale fading
from sim_core.params import SimParams
from sim_core.channel import ChannelModel
from sim_core.resource import ResourceBlockPool
from sim_core.helpers import dbm_to_linear, linear_to_dbm, linear_to_db

class BaseStation:
    def __init__(self, id, position, params: SimParams, channel_model: ChannelModel):
        self.id = id
        self.position = position
        self.params = params
        self.channel_model = channel_model
        self.power_per_link_beam_dbm = params.bs_tx_power_dbm
        self.load_factor_metric = 0.0 # From 0 to 1
        self.ues_served_this_step = set() # UEs for which this BS allocated RBs << 

    def get_access_beam_rsrp_at_ue(self, ue_pos):
        # This uses the calculate_path_loss method which includes path loss and shadowing
        path_loss_db = self.channel_model.calculate_path_loss(
            self.position, ue_pos
        )
        eirp_access_beam_dbm = (
            self.params.bs_tx_power_dbm + self.params.bs_access_beam_gain_db
        )
        return eirp_access_beam_dbm - path_loss_db

    def calculate_sinr_on_rb(self, target_ue, rb_id, all_bss_map, rb_pool: ResourceBlockPool):
        noise_w = self.channel_model.get_noise_power_linear(
            self.params.rb_bandwidth_hz
        )
        # Get the path loss (which includes shadowing)
        signal_path_loss_db = self.channel_model.calculate_path_loss(
            self.position, target_ue.position
        )
        signal_eirp_dbm = (
            self.power_per_link_beam_dbm + self.params.bs_link_beam_gain_db
        )
        received_signal_power_dbm = signal_eirp_dbm - signal_path_loss_db
        
        # --- NEW: Apply small-scale fading gain to desired signal ---
        # Rayleigh fading for amplitude, squared for power.
        # Mean of rayleigh.rvs(scale=sigma)^2 is 2*sigma^2.
        # To get a mean linear gain of 1 (0 dB), set sigma = 1/sqrt(2).
        # This models the instantaneous linear power gain due to small-scale fading.
        fading_gain_linear_signal = rayleigh.rvs(scale=1/np.sqrt(2))**2
        
        # Ensure it's not zero or extremely small which could lead to log errors later if converted to dB
        fading_gain_linear_signal = max(fading_gain_linear_signal, 1e-10) 

        signal_power_w = dbm_to_linear(received_signal_power_dbm) * fading_gain_linear_signal
        # --- END NEW ---

        interference_w = 0.0
        for other_bs_id, other_bs in all_bss_map.items():
            if other_bs.id == self.id:
                continue
            rb_alloc_info = rb_pool.rb_status.get(rb_id)
            if rb_alloc_info and rb_alloc_info["bs_id"] == other_bs.id:
                inter_path_loss_db = self.channel_model.calculate_path_loss(
                    other_bs.position, target_ue.position
                )
                inter_eirp_dbm = (
                    other_bs.power_per_link_beam_dbm
                    + self.params.bs_link_beam_gain_db
                )
                inter_power_w = dbm_to_linear(inter_eirp_dbm - inter_path_loss_db)
                
                # --- NEW: Apply small-scale fading gain to interference ---
                fading_gain_linear_interference = rayleigh.rvs(scale=1/np.sqrt(2))**2
                fading_gain_linear_interference = max(fading_gain_linear_interference, 1e-10)
                interference_w += inter_power_w * fading_gain_linear_interference
                # --- END NEW ---

        denominator = noise_w + interference_w
        if denominator <= 1e-22: # Prevent division by zero/very small number
            denominator = 1e-22
        sinr_linear = signal_power_w / denominator
        return max(sinr_linear, dbm_to_linear(-20)) # Cap minimum SINR

    def update_load_factor(self, ues_map):
        connected_ues_to_this_bs = [
            ue
            for ue in ues_map.values()
            if (ue.serving_bs_1 == self or ue.serving_bs_2 == self)
            and ue.current_total_rate_mbps > 0 # Only count UEs effectively served
        ]

        # Sum of rates *provided by this BS* (simplified approximation)
        theta_j_t = sum(
            ue.get_rate_from_bs(self.id) for ue in connected_ues_to_this_bs
        )
        u_j_m_t = len(connected_ues_to_this_bs)

        if u_j_m_t > 0:
            avg_cap_per_ue_from_this_bs_mbps = theta_j_t / u_j_m_t
            # Load factor definition is subjective. Here, it relates to target rate.
            # 1.0 if near target, 0.75 if above target, 0.25 if below target.
            if (
                abs(avg_cap_per_ue_from_this_bs_mbps - self.params.target_ue_throughput_mbps)
                < 0.2 * self.params.target_ue_throughput_mbps
            ):
                self.load_factor_metric = 1.0
            elif avg_cap_per_ue_from_this_bs_mbps > self.params.target_ue_throughput_mbps:
                self.load_factor_metric = 0.75
            else:
                self.load_factor_metric = 0.25
        else:
            self.load_factor_metric = 0.0 # No load or no UEs served effectively


class UserEquipment:
    def __init__(self, id, initial_pos, params: SimParams):
        self.id = id
        self.position = np.array(initial_pos, dtype=float)
        self.params = params
        self.serving_bs_1: BaseStation = None
        self.serving_bs_2: BaseStation = None
        self.measurements = {} # {bs_id: rsrp_dbm}
        self.best_bs_candidates = [] # [(bs_id, rsrp)] sorted by rsrp DESC

        self.current_total_rate_mbps = 0.0
        self.sinr_db_bs1 = -np.inf
        self.sinr_db_bs2 = -np.inf
        self.rbs_from_bs1 = []
        self.rbs_from_bs2 = []

        self.time_in_a3_condition = {} # For baseline handover

    def move(self):
        angle = np.random.uniform(0, 2 * np.pi)
        distance = self.params.ue_speed_mps * self.params.time_step_duration
        self.position[0] += distance * np.cos(angle)
        self.position[1] += distance * np.sin(angle)
        self.position[0] = np.clip(self.position[0], 0, self.params.sim_area_x)
        self.position[1] = np.clip(self.position[1], 0, self.params.sim_area_y)

    def perform_measurements(self, all_bss_map):
        self.measurements.clear()
        self.best_bs_candidates.clear()
        temp_measurements = []
        for bs_id, bs in all_bss_map.items():
            try:
                rsrp = bs.get_access_beam_rsrp_at_ue(self.position)
                self.measurements[bs_id] = rsrp
                # Only consider BSs above a certain threshold for candidates to save computation
                if rsrp > self.params.min_rsrp_for_acq_dbm - 20: # Example threshold
                    temp_measurements.append((bs_id, rsrp))
            except Exception: # Handle potential issues like a BS not having a valid position
                self.measurements[bs_id] = -np.inf
        self.best_bs_candidates = sorted(
            temp_measurements, key=lambda item: item[1], reverse=True
        )

    def reset_connections_and_rates(self):
        self.serving_bs_1 = None
        self.serving_bs_2 = None
        self.current_total_rate_mbps = 0.0
        self.sinr_db_bs1 = -np.inf
        self.sinr_db_bs2 = -np.inf
        self.rbs_from_bs1.clear()
        self.rbs_from_bs2.clear()

    def get_rate_from_bs(self, bs_id_query):
        # This is a simplified way to approximate rate from a specific BS for load calculation.
        # It assumes that if a UE is dual-connected, its total rate is roughly split between BSs.
        if (
            self.serving_bs_1
            and self.serving_bs_1.id == bs_id_query
        ):
            # If dual connected, approximate as half, otherwise all
            return (
                self.current_total_rate_mbps / 2
                if self.serving_bs_2 and self.serving_bs_2 != self.serving_bs_1
                else self.current_total_rate_mbps
            )
        elif self.serving_bs_2 and self.serving_bs_2.id == bs_id_query:
            # Must be dual connected if BS2 is serving and is not BS1
            return self.current_total_rate_mbps / 2
        return 0.0

    def update_ue_rates_sinrs(self, all_bss_map, rb_pool: ResourceBlockPool):
        self.current_total_rate_mbps = 0.0
        rate_bs1_bps = 0.0
        sinrs_bs1_linear = []
        if self.serving_bs_1 and self.rbs_from_bs1:
            for rb_id in self.rbs_from_bs1:
                # Ensure the RB is still allocated to this UE by this BS
                if (
                    rb_pool.rb_status.get(rb_id, {}).get("ue_id") == self.id
                    and rb_pool.rb_status.get(rb_id, {}).get("bs_id")
                    == self.serving_bs_1.id
                ):
                    sinr_l = self.serving_bs_1.calculate_sinr_on_rb(
                        self, rb_id, all_bss_map, rb_pool
                    )
                    rate_bs1_bps += self.params.channel_model.shannon_capacity(
                        sinr_l, self.params.rb_bandwidth_hz
                    )
                    sinrs_bs1_linear.append(sinr_l)
            self.sinr_db_bs1 = (
                linear_to_db(np.mean(sinrs_bs1_linear))
                if sinrs_bs1_linear
                else -np.inf
            )
        else:
            self.sinr_db_bs1 = -np.inf

        rate_bs2_bps = 0.0
        sinrs_bs2_linear = []
        if self.serving_bs_2 and self.rbs_from_bs2:
            for rb_id in self.rbs_from_bs2:
                if (
                    rb_pool.rb_status.get(rb_id, {}).get("ue_id") == self.id
                    and rb_pool.rb_status.get(rb_id, {}).get("bs_id")
                    == self.serving_bs_2.id
                ):
                    sinr_l = self.serving_bs_2.calculate_sinr_on_rb(
                        self, rb_id, all_bss_map, rb_pool
                    )
                    rate_bs2_bps += self.params.channel_model.shannon_capacity(
                        sinr_l, self.params.rb_bandwidth_hz
                    )
                    sinrs_bs2_linear.append(sinr_l)
            self.sinr_db_bs2 = (
                linear_to_db(np.mean(sinrs_bs2_linear))
                if sinrs_bs2_linear
                else -np.inf
            )
        else:
            self.sinr_db_bs2 = -np.inf

        self.current_total_rate_mbps = (rate_bs1_bps + rate_bs2_bps) / 1e6