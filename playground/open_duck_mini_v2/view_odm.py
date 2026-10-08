from mujoco import MjModel, MjData, mj_step, mj_forward
from mujoco.viewer import launch_passive
import os
import time

XML = os.path.join(os.path.dirname(__file__), "xmls", "open_duck_mini_v2.xml")

model = MjModel.from_xml_path(XML)
data = MjData(model)

model.vis.global_.offwidth  = 960
model.vis.global_.offheight = 540
model.vis.quality.offsamples = 0
model.vis.quality.shadowsize = 256  # try 0, 256, 512
model.opt.timestep = 0.002          # preview-friendly

SUBSTEPS_PER_FRAME = 2  # keep small; increase if sim looks too fast

viewer = launch_passive(model, data)
while viewer.is_running():
    step_start = time.time()
    mj_step(model, data)



