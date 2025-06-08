BOLTZMANN = 1.380649e-23
TEMP_KELVIN = 290.0

# Try to import TensorFlow and set availability
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False 