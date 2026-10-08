# playground/export_shims.py
import jax.numpy as jnp
from flax import linen as nn

class PolicyMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(256)(x))
        x = nn.tanh(nn.Dense(256)(x))
        return nn.Dense(ACTION_DIM)(x)

OBS_DIM = 123
ACTION_DIM = 10

def build(key=None):
    import jax
    model = PolicyMLP()
    key = key or jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, OBS_DIM), jnp.float32)
    init_vars = model.init(key, dummy)
    init_params = init_vars["params"]

    def apply_fn(params, obs):
        return model.apply({"params": params}, obs, mutable=False)  # no 'train='

    return apply_fn, init_params, OBS_DIM

# same model as above
OBS_DIM = 123

def build_policy_apply():
    model = PolicyMLP()
    def apply_fn(params, obs):
        return model.apply({"params": params}, obs, mutable=False)
    return apply_fn

def build_init_params(key):
    dummy = jnp.zeros((1, OBS_DIM), jnp.float32)
    return PolicyMLP().init(key, dummy)["params"]
