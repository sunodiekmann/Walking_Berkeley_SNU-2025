import warnings, numpy as np, jax, jax.numpy as jp
warnings.filterwarnings("error", message="overflow encountered in cast")
np.seterr(over='raise')  # make NumPy raise on overflow

from playground.berkeley_humanoid_lite.joystick import Joystick

env = Joystick(task="flat_terrain")
key = jax.random.PRNGKey(0)
state = env.reset(key)  # forces first obs
# force materialization & basic stats:
s = state.obs["state"]
print("shape:", s.shape, "finite:", jp.isfinite(s).all(), "max|s|:", jp.max(jp.abs(s)))
