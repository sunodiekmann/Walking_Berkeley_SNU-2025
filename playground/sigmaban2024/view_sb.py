import time
import numpy as np
import mujoco
from mujoco import MjModel, MjData, mj_step
from mujoco.viewer import launch_passive, launch

SCENE = "sigmaban2024/xmls/scene_flat_terrain.xml"

model = MjModel.from_xml_path(SCENE)
data = MjData(model)

# data.qpos[model.joint("arm_left_shoulder_pitch_joint").qposadr[0]]   = -0.5
# data.qpos[model.joint("arm_right_shoulder_pitch_joint").qposadr[0]]  = 0.5
# data.qpos[model.joint("leg_left_hip_pitch_joint").qposadr[0]]   = -0.5
# data.qpos[model.joint("leg_right_hip_pitch_joint").qposadr[0]]  = -0.5
# data.qpos[model.joint("leg_left_knee_pitch_joint").qposadr[0]]  =  1.2
# data.qpos[model.joint("leg_right_knee_pitch_joint").qposadr[0]] =  1.2
# data.qpos[model.joint("leg_left_ankle_pitch_joint").qposadr[0]] = -0.7
# data.qpos[model.joint("leg_right_ankle_pitch_joint").qposadr[0]] = -0.7
data.qpos[:] = model.keyframe("home").qpos
data.qvel[:] = 0.0

mujoco.mj_forward(model, data)

for i in range(model.nu):
    j_id = model.actuator(i).trnid[0]             # joint id driven by actuator i
    qadr = model.jnt_qposadr[j_id]                # that joint's qpos address
    data.ctrl[i] = data.qpos[qadr]  


print("qpos:", " ".join([f"{x:.6f}" for x in data.qpos]))

# 6) preview
viewer = launch(model, data)
while viewer.is_running():
    mj_step(model, data)


