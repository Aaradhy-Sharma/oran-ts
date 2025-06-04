import numpy as np
import math

# type: def dbm_to_linear(dbm) -> float
def dbm_to_linear(dbm):
    if dbm == -np.inf or math.isinf(dbm) or math.isnan(dbm):
        return 0.0
    return 10 ** ((dbm - 30) / 10)

# type: def linear_to_dbm(linear) -> float
def linear_to_dbm(linear):
    if linear <= 0 or math.isinf(linear) or math.isnan(linear):
        return -np.inf
    return 10 * np.log10(linear) + 30

# type: def db_to_linear(db) -> float
def db_to_linear(db):
    if math.isinf(db) or math.isnan(db):
        return 0.0
    return 10 ** (db / 10)

# type: def linear_to_db(linear) -> float
def linear_to_db(linear):
    if linear <= 0 or math.isinf(linear) or math.isnan(linear):
        return -np.inf
    return 10 * np.log10(linear) + 30