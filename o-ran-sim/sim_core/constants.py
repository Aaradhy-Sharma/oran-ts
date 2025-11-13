BOLTZMANN = 1.380649e-23
TEMP_KELVIN = 290.0
try:
	# Try importing TensorFlow to detect availability at import-time.
	import tensorflow  # type: ignore
	TF_AVAILABLE = True
except Exception:
	TF_AVAILABLE = False