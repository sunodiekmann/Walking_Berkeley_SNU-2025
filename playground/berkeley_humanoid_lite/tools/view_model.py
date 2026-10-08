import mujoco
from mujoco import MjModel, MjData
from mujoco.viewer import launch_passive

from playground.berkeley_humanoid_lite import constants

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
model = MjModel.from_xml_path(str(constants.FLAT_TERRAIN_XML))
data = MjData(model)

# ------------------------------------------------------------------
# Reset to "home" keyframe
# ------------------------------------------------------------------
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
assert key_id >= 0, "Keyframe 'home' not found"

data.qpos[:] = model.key_qpos[key_id]
data.qvel[:] = 0.0

mujoco.mj_forward(model, data)

# ------------------------------------------------------------------
# Launch viewer
# ------------------------------------------------------------------
viewer = launch_passive(model, data)

while viewer.is_running():
    mujoco.mj_step(model, data)
