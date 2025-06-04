import numpy as np
import math
from sim_core.params import SimParams
from sim_core.constants import BOLTZMANN, TEMP_KELVIN
from sim_core.helpers import db_to_linear

# type: class ChannelModel(object)
class ChannelModel:
    def __init__(self, params: SimParams):
        self.params = params

    def calculate_path_loss(self, pos1, pos2):
        dist = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        dist = max(dist, 1e-1) # Avoid log(0)
        if dist <= self.params.ref_dist_m:
            path_loss = self.params.ref_loss_db
        else:
            path_loss = (
                self.params.ref_loss_db
                + 10
                * self.params.path_loss_exponent
                * np.log10(dist / self.params.ref_dist_m)
            )
        if self.params.shadowing_std_dev_db > 0:
            path_loss += np.random.normal(0, self.params.shadowing_std_dev_db)
        return path_loss

    def get_noise_power_linear(self, bandwidth_hz):
        return (
            BOLTZMANN
            * TEMP_KELVIN
            * db_to_linear(self.params.ue_noise_figure_db)
            * bandwidth_hz
        )

    def shannon_capacity(self, sinr_linear, bandwidth_hz):
        if sinr_linear <= -1 or math.isinf(sinr_linear) or math.isnan(sinr_linear):
            return 0.0
        capped_sinr = min(sinr_linear, db_to_linear(30)) # Cap SINR to avoid unrealistic capacity
        return bandwidth_hz * np.log2(1 + capped_sinr)