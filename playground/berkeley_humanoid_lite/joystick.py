"""Joystick task for BHL (no head/neck)."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from jax import debug as jdebug
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import constants
from . import base as bhl_base

from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.common.rewards import (
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    cost_torques,
    cost_action_rate,
    reward_alive,
)
from playground.berkeley_humanoid_lite.custom_rewards import (
    reward_imitation,
)

# Toggles (can be moved into config if desired)
USE_IMITATION_REWARD = True
USE_MOTOR_SPEED_LIMITS = True


def default_config() -> config_dict.ConfigDict:
  # All literals are Python floats; with JAX x64 disabled they become float32.
  return config_dict.create(
      # Time / control
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.25,
      dof_vel_scale=0.05,
      history_len=0,
      soft_joint_pos_limit_factor=0.95,
      max_motor_velocity=5.24, #5.24,  # rad/s

      # Non-uniform command horizons will be sampled over [2, 10] s
      command_horizon_sec_range=[2.0, 10.0],

      accelerometer_bias_x=0.0,
      base_height_nominal=0.675,

      term_height_min=0.35,         # m (pelvis / root min height)
      term_max_tilt_deg=55.0,       # degrees (terminate if tilt beyond this)

      noise_config=config_dict.create(
          level=0.0,
          action_min_delay=0,  # env steps
          action_max_delay=3,  # env steps
          imu_min_delay=0,     # env steps
          imu_max_delay=3,     # env steps
          scales=config_dict.create(
              hip_pos=0.03,    # rad
              knee_pos=0.05,   # rad
              ankle_pos=0.08,  # rad
              joint_vel=2.5,   # rad/s
              gravity=0.1,
              linvel=0.1,
              gyro=0.1,
              accelerometer=0.05,
          ),
      ),

      reward_config=config_dict.create(
          scales=config_dict.create(
              imitation        = 7.0,  # start: 6.0
              tracking_lin_vel = 4.0,      # start: 4.0
              tracking_ang_vel = 4.0,      # start: 5.0
              upright          = 1.0,      # start: 1.0

              torques     = -1e-7,       # stronger than -2e-5, but still mild
              action_rate = -0.0005,         # penalize chattering

              foot_slip        = -0.5,     # start: -0.5
              step_distance    = 25.0,     # start: 20.0
              foot_clearance   = 25.0,     # start: 20.0
              feet_air_time    = 25.0,     #  start: 20.0
              foot_seperation  = 0.5,      #  start: 0.5
              swing_progress   = 2.5,      # start: 1.5

              knee_flexion     = 6.0,      #  start: 2.5
              stillness        = -1.5,     # penalty for doing something when should be still
              lazyness         = -1.5,     # penalty for doing nothing when should move
              yaw_drift        = -0.3,     # don’t drift sideways when commanded straight
              foot_flat        = 0.1,

              # Episode-level
              alive            = 0.5,
              termination      = -100.0,
          ),
          tracking_sigma=0.015, 

          foot_clearance_target=0.04,   # unused if foot_clearance=0 but ok
          feet_air_time_threshold=0.4,  # unused if feet_air_time=0 but ok
      ),

      # Pushes disabled for now
      push_config=config_dict.create(
          enable=False,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 1.0],
          warmup_steps=100,  # steps before pushes are allowed
      ),
      period = 1.02,
      lin_vel_x = [-0.115, -0.058, 0.0, 0.058, 0.115, 0.173],
      lin_vel_y = [-0.058, 0.0, 0.058],
      ang_vel_yaw = [-0.385, -0.192, 0.0, 0.192, 0.385],
  )


class Joystick(bhl_base.BHLEnv):
  """Track a joystick command (vx, vy, yaw_rate) with locomotion shaping."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
      super().__init__(
          xml_path=constants.task_to_xml(task).as_posix(),
          config=config,
          config_overrides=config_overrides,
      )
      self._post_init()

  # -------------------------
  # Helpers
  # -------------------------

  def _compute_motor_targets(
      self,
      prev_motor_targets: jax.Array,
      action_command: jax.Array,
  ) -> jax.Array:
      """Absolute pose target around neutral + per-step slew limiting."""
      proposed = self._default_actuator + action_command * self._config.action_scale
      delta = proposed - prev_motor_targets
      if USE_MOTOR_SPEED_LIMITS:
          max_step = jp.asarray(self._config.max_motor_velocity, jp.float32) * self.dt
          delta = jp.clip(delta, -max_step, max_step)
      return prev_motor_targets + delta


  def cost_lazy(
      self,
      liftoff_feet_body: jax.Array,     # (nfeet,3)
      curr_feet_body: jax.Array,        # (nfeet,3)
      first_contact: jax.Array,             # (nfeet,) bool
      cmd_speed: jax.Array,             # scalar
  ) -> jax.Array:
      # measure horizontal step length on touchdown
      dxy = curr_feet_body[:, :2] - liftoff_feet_body[:, :2]
      step_len = jp.linalg.norm(dxy, axis=-1)  # (nfeet,)
  
      # choose a minimum step length; scale slightly with command speed
      min_step = jp.asarray(0.03, jp.float32) * jp.clip(cmd_speed / 0.2, 0.5, 2.0)
      min_step = jp.clip(min_step, 0.02, 0.14)  # keep within your expected range
  
      # cost only on touchdown feet
      td = first_contact.astype(step_len.dtype)
      shortfall = jp.clip(min_step - step_len, 0.0, min_step) / jp.maximum(min_step, 1e-6)
      # average over touchdown feet; if none, cost 0
      denom = jp.maximum(jp.sum(td), 1.0)
      return jp.sum(shortfall * td) / denom

  

  def cost_foot_slip(self, feet_vel_body: jax.Array, contact_phase: jax.Array) -> jax.Array:
      """
      Penalize horizontal foot speed when in contact.
      feet_vel_body: (nfeet,3) relative velocity in base frame (already corrected with -ω×p)
      contact_phase: (nfeet,) bool or {0,1}
      """
      v_xy = jp.linalg.norm(feet_vel_body[:, :2], axis=-1)  # (nfeet,)
      c = contact_phase.astype(v_xy.dtype)
      val = jp.sum(v_xy * c)
      return jp.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

  
  def reward_upright(self, data: mjx.Data) -> jax.Array:
      """Reward torso pitch near -8° (forward lean)."""
      R_flat = data.site_xmat[self._site_id]      # (9,)
      R = R_flat.reshape(3, 3)                    # (3,3)
  
      pitch = jp.arctan2(-R[2, 0], R[2, 2])       # rad
  
      pitch_target = jp.deg2rad(jp.asarray(-8.0, jp.float32))
      sigma = jp.deg2rad(jp.asarray(5.0, jp.float32))  # ±5° band
  
      err = (pitch - pitch_target) / jp.maximum(sigma, 1e-6)
      val = jp.exp(-0.5 * err**2)
  
      return jp.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

  def reward_foot_separation(
      self,
      feet_pos_body: jax.Array,  # (2,3) in base frame
      command: jax.Array,        # (3,) [vx, vy, yaw_rate]
      target: float = 0.16,
  ) -> jax.Array:
      cmd_speed = jp.linalg.norm(command[:2])
      moving = cmd_speed > 0.05
  
      # lateral distance in body frame
      d = jp.abs(feet_pos_body[0, 1] - feet_pos_body[1, 1])
  
      target = jp.asarray(target, jp.float32)        # 0.16
      sigma = jp.asarray(0.05, jp.float32)           # ~3–5 cm tolerance feels reasonable
  
      # Gaussian around target
      err = (d - target) / jp.maximum(sigma, 1e-6)
      shape = jp.exp(-0.5 * err**2)
  
      # mild penalty if extremely narrow (encourage not crossing)
      min_w = jp.asarray(0.10, jp.float32)
      narrow = jp.clip((min_w - d) / jp.maximum(min_w, 1e-6), 0.0, 1.0)  # 0..1
      penalty = 0.5 * narrow  # mild
  
      val = shape - penalty
      val = jp.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
      return jp.where(moving, val, jp.array(0.0, dtype=jp.float32))
  


  def reward_foot_clearance_touchdown(
      self,
      swing_peak_z: jax.Array,     # (nfeet,) peak z during swing (prefer BODY z)
      first_contact: jax.Array,    # (nfeet,)
      command: jax.Array,
      z_target: float = 0.04,
  ) -> jax.Array:
      cmd_speed = jp.linalg.norm(command[:2])
      moving = cmd_speed > 0.05
  
      z_target = jp.asarray(z_target, jp.float32)
  
      # soft score around target
      sigma = jp.asarray(0.03, jp.float32)  # looser early than 0.015
      z_err = (swing_peak_z - z_target) / jp.maximum(sigma, 1e-6)
      score = jp.exp(-0.5 * z_err**2)
  
      # "hit" bonus (binary) to discourage low shuffles
      hit = (swing_peak_z >= z_target).astype(jp.float32)
  
      td = first_contact.astype(jp.float32)
      val = jp.sum(td * (0.7 * score + 0.3 * hit))
  
      return jp.where(moving, jp.nan_to_num(val), jp.array(0.0, jp.float32))


  def reward_feet_air_time_touchdown(
      self,
      feet_air_time: jax.Array,        # (nfeet,) BEFORE reset
      first_contact: jax.Array,        # (nfeet,)
      swing_peak_z: jax.Array,         # (nfeet,) (prefer BODY z)
      command: jax.Array,
      target_air_time: float = 0.40,
  ) -> jax.Array:
      cmd_speed = jp.linalg.norm(command[:2])
      moving = cmd_speed > 0.05
  
      target = jp.asarray(target_air_time, jp.float32)
      sigma = jp.asarray(0.20, jp.float32)
  
      err = (feet_air_time - target) / jp.maximum(sigma, 1e-6)
      score = jp.exp(-0.5 * err**2)
  
      # soft validity weights instead of hard thresholds
      w_air  = jp.clip((feet_air_time - 0.05) / 0.10, 0.0, 1.0)     # ramps up from 0.05s to 0.15s
      w_peak = jp.clip((swing_peak_z - 0.005) / 0.02, 0.0, 1.0)     # ramps up from 5mm to 25mm
  
      td = first_contact.astype(jp.float32)
      val = jp.sum(td * score * w_air * w_peak)
  
      return jp.where(moving, jp.nan_to_num(val), jp.array(0.0, jp.float32))


  def reward_knee_flexion(
      self,
      joint_angles: jax.Array,   # (n_joints,)
      command: jax.Array,        # (3,)
  ) -> jax.Array:
    #   cmd_speed = jp.linalg.norm(command[:2])
    #   moving = cmd_speed > 0.05
  
      knees = joint_angles[self._knee_ids]  # (2,)
  
      # Prefer knees in a band: [min_flex, max_flex]
      min_flex = jp.asarray(0.75, jp.float32)   # ~20°
      max_flex = jp.asarray(1.25, jp.float32)   # ~63°
  
      # 1 inside band, smoothly decays outside
      below = jp.clip((min_flex - knees) / jp.maximum(min_flex, 1e-6), 0.0, 1.0)
      above = jp.clip((knees - max_flex) / jp.maximum(max_flex, 1e-6), 0.0, 1.0)
      score = jp.exp(-2.0 * (below**2 + above**2))  # (2,)
      val = jp.mean(score)
  
      val = jp.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
      return val

  def reward_step_distance_touchdown(
      self,
      liftoff_feet_body: jax.Array,     # (nfeet,3)
      curr_feet_body: jax.Array,        # (nfeet,3)
      command: jax.Array,               # (3,) [vx, vy, yaw_rate] in BODY
      first_contact: jax.Array,         # (nfeet,) RAW touchdown
      feet_air_time: jax.Array,         # (nfeet,) BEFORE reset
      swing_peak_z: jax.Array,     # (nfeet,) BODY-frame peak z
  ) -> jax.Array:
  
      cmd_xy = command[:2]
      cmd_speed = jp.linalg.norm(cmd_xy)
      moving = cmd_speed > jp.asarray(0.05, jp.float32)
  
      cmd_dir = jp.where(
          cmd_speed > 1e-6,
          cmd_xy / cmd_speed,
          jp.array([1.0, 0.0], jp.float32),
      )
  
      step_vec = curr_feet_body[:, :2] - liftoff_feet_body[:, :2]
      step_len = jp.linalg.norm(step_vec, axis=-1)
  
      step_dir = step_vec / jp.maximum(step_len[:, None], 1e-6)
      align = jp.clip(jp.sum(step_dir * cmd_dir[None, :], axis=-1), 0.0, 1.0)
  
      # validity gates
      min_air  = jp.asarray(0.10, jp.float32)
      min_peak = jp.asarray(0.02, jp.float32)
      valid = (feet_air_time > min_air) & (swing_peak_z > min_peak)
  
      td = first_contact.astype(jp.float32) * valid.astype(jp.float32)
  
      # step length band
      min_step = jp.asarray(0.03, jp.float32)
      max_step = jp.asarray(0.14, jp.float32)
      too_small = jp.clip(min_step - step_len, 0.0, None) / min_step
      too_big   = jp.clip(step_len - max_step, 0.0, None) / max_step
      band = jp.exp(-2.0 * (too_small**2 + too_big**2))
  
      val = jp.sum(td * align * band)
      val = jp.nan_to_num(val)
  
      return jp.where(moving, val, jp.array(0.0, jp.float32))
  


  def reward_swing_progress_dense(
      self,
      feet_vel_body: jax.Array,     # (nfeet,3)
      contact_phase: jax.Array,     # (nfeet,) bool (filtered)
      command: jax.Array,           # (3,)
  ) -> jax.Array:
      contact_phase = contact_phase.astype(jp.bool_)
  
      cmd_xy = command[:2]
      cmd_speed = jp.linalg.norm(cmd_xy)
      moving = cmd_speed > jp.asarray(0.05, jp.float32)
  
      cmd_dir = jp.where(
          cmd_speed > 1e-6,
          cmd_xy / cmd_speed,
          jp.array([1.0, 0.0], jp.float32),
      )
  
      swing = (~contact_phase).astype(jp.float32)  # 1 swing
      vxy = feet_vel_body[:, :2]
  
      # reward forward velocity during swing; cap contribution
      forward = jp.clip(jp.sum(vxy * cmd_dir[None, :], axis=-1), 0.0, 1.0)
      val = jp.sum(forward * swing)
  
      val = jp.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
      return jp.where(moving, val, jp.array(0.0, jp.float32))
  
  def cost_foot_flat_contact(
      self,
      data: mjx.Data,
      contact_phase: jax.Array,          # (nfeet,) bool or {0,1}
      feet_site_id: jax.Array,           # (nfeet,) int
      k: float = 10.0,                   # sharpness for exp form (optional)
      use_exp: bool = False,
  ) -> jax.Array:
      """Penalize foot tilt during contact; yaw-invariant."""
      # World-up
      z_w = jp.array([0.0, 0.0, 1.0], dtype=jp.float32)
  
      def one_foot(i):
          R_wf = jp.reshape(data.site_xmat[feet_site_id[i]], (3, 3)).astype(jp.float32)
          n_w = R_wf[:, 2]  # foot local +Z expressed in world
          align = jp.clip(jp.dot(n_w, z_w), -1.0, 1.0)  # 1 when perfectly flat/upright
  
          # Tilt measure (0 is best)
          tilt = 1.0 - align
  
          # Contact gate (only care when foot is supporting)
          c = contact_phase[i].astype(jp.float32)
  
          if use_exp:
              # bounded [0,1], smoother; convert to cost
              return c * (1.0 - jp.exp(-k * tilt))
          else:
              # simple linear cost
              return c * tilt
  
      costs = jp.stack([one_foot(i) for i in range(feet_site_id.shape[0])], axis=0)
      return jp.sum(costs)
  
  
  def _sample_cmd_horizon_steps(self, rng: jax.Array) -> jax.Array:
      """Sample command horizon with a non-uniform mixture over [2, 10] seconds."""
      u = jax.random.uniform(rng, dtype=jp.float32)
  
      short = u < 0.3
      med   = (u >= 0.3) & (u < 0.7)
      long  = u >= 0.7

      h_short = 2.0 + (u / 0.3) * (4.0 - 2.0)          # 2–4
      h_med   = 4.0 + ((u - 0.3) / 0.4) * (7.0 - 4.0)  # 4–7
      h_long  = 7.0 + ((u - 0.7) / 0.3) * (10.0 - 7.0) # 7–10

      horizon_sec = jp.where(short, h_short, jp.where(med, h_med, h_long))
      steps = jp.round(h_long / self.dt).astype(jp.int32)
      return steps

  # -------------------------
  # Init
  # -------------------------

  def _post_init(self) -> None:
      self._init_q = jp.asarray(self._mj_model.keyframe("home").qpos, dtype=jp.float32)
      self._default_actuator = jp.asarray(
          self._mj_model.keyframe("home").ctrl, dtype=jp.float32
      )

      # Joint limits (skip floating base joint 0).
      jnt_range = jp.asarray(self.mj_model.jnt_range[1:], dtype=jp.float32)
      self._lowers, self._uppers = jnt_range.T
      c = (self._lowers + self._uppers) / 2.0
      r = self._uppers - self._lowers
      factor = jp.asarray(self._config.soft_joint_pos_limit_factor, jp.float32)
      self._soft_lowers = c - 0.5 * r * factor
      self._soft_uppers = c + 0.5 * r * factor

      # Map soft limits to actuated joints.
      act_idx = np.array(self.actuator_joint_ids) - 1
      self._soft_lowers_act = jp.asarray(
          np.take(np.array(self._soft_lowers), act_idx), dtype=jp.float32
      )
      self._soft_uppers_act = jp.asarray(
          np.take(np.array(self._soft_uppers), act_idx), dtype=jp.float32
      )

      self._njoints = self._mj_model.njnt
      self._actuators = self._mj_model.nu

      self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
      self._torso_mass = jp.asarray(
          self._mj_model.body_subtreemass[self._torso_body_id], dtype=jp.float32
      )
      self._site_id = self._mj_model.site("imu").id  # IMU frame

      self._feet_site_id = np.array(
          [self._mj_model.site(name).id for name in constants.FEET_SITES],
          dtype=np.int32,
      )
      self._floor_geom_id = self._mj_model.geom("floor").id
      self._feet_geom_id = np.array(
          [self._mj_model.geom(name).id for name in constants.FEET_GEOMS],
          dtype=np.int32,
      )
      
      # --- Foot world position sensor addresses per foot ---
      foot_pos_sensor_adr = []
      for site in constants.FEET_SITES:
          sensor_id = self._mj_model.sensor(f"{site}_pos").id
          sensor_adr = self._mj_model.sensor_adr[sensor_id]
          sensor_dim = self._mj_model.sensor_dim[sensor_id]
          assert sensor_dim == 3, f"{site}_pos sensor dim expected 3, got {sensor_dim}"
          foot_pos_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
      self._foot_pos_sensor_adr = jp.asarray(foot_pos_sensor_adr, dtype=jp.int32)
  
      # --- Foot world linear velocity sensor addresses per foot ---
      foot_linvel_sensor_adr = []
      for site in constants.FEET_SITES:
          sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
          sensor_adr = self._mj_model.sensor_adr[sensor_id]
          sensor_dim = self._mj_model.sensor_dim[sensor_id]
          assert sensor_dim == 3, f"{site}_global_linvel sensor dim expected 3, got {sensor_dim}"
          foot_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
      self._foot_linvel_sensor_adr = jp.asarray(foot_linvel_sensor_adr, dtype=jp.int32)

      # Joint QPOS noise scales per actuator (hips/knees/ankles).
      qpos_noise_scale = np.zeros(self._actuators, dtype=np.float32)
      hip_ids = [idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_hip" in j]
      knee_ids = [idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_knee" in j]
      ankle_ids = [idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_ankle" in j]

      self._hip_ids = jp.asarray(hip_ids, dtype=jp.int32)
      self._knee_ids = jp.asarray(knee_ids, dtype=jp.int32)
      self._ankle_ids = jp.asarray(ankle_ids, dtype=jp.int32)

      qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
      qpos_noise_scale[knee_ids] = self._config.noise_config.scales.knee_pos
      qpos_noise_scale[ankle_ids] = self._config.noise_config.scales.ankle_pos
      self._qpos_noise_scale = jp.asarray(qpos_noise_scale, dtype=jp.float32)

      self._nfeet = len(constants.FEET_SITES)
      self._base_body_id = self._mj_model.body("base").id
      self._imu_site_id = self._mj_model.site("imu").id


      # Imitation reference motion
      if USE_IMITATION_REWARD:
          self.PRM = PolyReferenceMotion(str(constants.REFERENCE_MOTION_PKL))
          example_ref = jp.asarray(
              self.PRM.get_reference_motion(0.0, 0.0, 0.0, 0), dtype=jp.float32
          )
          self._ref_dim = int(example_ref.shape[0])

  # -------------------------
  # Reset
  # -------------------------

  def reset(self, rng: jax.Array) -> mjx_env.State:
      qpos = self._init_q
      qvel = jp.zeros(self.mjx_model.nv, dtype=jp.float32)
  
      # Randomize XY spawn and yaw.
      rng, key = jax.random.split(rng)
      dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05, dtype=jp.float32)
  
      base_qpos = self.get_floating_base_qpos(qpos)
      base_qpos = base_qpos.at[0:2].set(
          qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2] + dxy
      )
  
      rng, key = jax.random.split(rng)
      yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14, dtype=jp.float32)
      quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0], dtype=jp.float32), yaw)
      new_quat = math.quat_mul(
          qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7],
          quat,
      )
      base_qpos = base_qpos.at[3:7].set(new_quat)
      qpos = self.set_floating_base_qpos(base_qpos, qpos)
  
      # Add small additive noise to actuated joints.
      rng, key = jax.random.split(rng)
      qpos_j = (
          self.get_actuator_joints_qpos(qpos)
          + jax.random.normal(key, (self._actuators,), dtype=jp.float32) * 0.05
      )
      qpos = self.set_actuator_joints_qpos(qpos_j, qpos)
  
      # Randomize floating base velocity a bit.
      rng, key = jax.random.split(rng)
      qvel = self.set_floating_base_qvel(
          jax.random.uniform(key, (6,), minval=-0.05, maxval=0.05, dtype=jp.float32),
          qvel,
      )
  
      ctrl = self.get_actuator_joints_qpos(qpos).astype(jp.float32)
      data = self._mjx_data0.replace(
          qpos=qpos.astype(jp.float32),
          qvel=qvel.astype(jp.float32),
          ctrl=ctrl,
      )
      data = mjx.forward(self.mjx_model, data)

      # Initial command + randomized horizon
      rng, cmd_rng, hor_rng = jax.random.split(rng, 3)
      cmd = self.sample_command(cmd_rng)  # [vx, vy, yaw_rate]
      cmd_horizon_steps = self._sample_cmd_horizon_steps(hor_rng)
  
      # Push interval sampling (no-op because enable=False)
      rng, push_rng = jax.random.split(rng)
      push_interval = jax.random.uniform(
          push_rng,
          minval=self._config.push_config.interval_range[0],
          maxval=self._config.push_config.interval_range[1],
          dtype=jp.float32,
      )
      push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)
  
      # Imitation reference motion
      if USE_IMITATION_REWARD:
          ref = self.PRM.get_reference_motion(cmd[0], cmd[1], cmd[2], 0)
          current_reference_motion = jp.asarray(ref[: self._ref_dim], dtype=jp.float32)
      else:
          current_reference_motion = jp.zeros(0, dtype=jp.float32)
  
      # Info dict 
      info = {
          "rng": rng,
          "step": jp.array(0, dtype=jp.int32),
          "command": cmd,
          "cmd_step": jp.array(0, dtype=jp.int32),
          "cmd_horizon_steps": cmd_horizon_steps,
          "last_act": jp.zeros(self.mjx_model.nu, dtype=jp.float32),
          "last_last_act": jp.zeros(self.mjx_model.nu, dtype=jp.float32),
          "last_last_last_act": jp.zeros(self.mjx_model.nu, dtype=jp.float32),
          "motor_targets": ctrl,
  
          # Push related.
          "push": jp.zeros(2, dtype=jp.float32),
          "push_step": jp.array(0, dtype=jp.int32),
          "push_interval_steps": push_interval_steps,
  
          # History related.
          "action_history": jp.zeros(
              self._config.noise_config.action_max_delay * self._actuators,
              dtype=jp.float32,
          ),
          "imu_history": jp.zeros(
              self._config.noise_config.imu_max_delay * 3, dtype=jp.float32
          ),
  
          # Imitation related.
          "imitation_i": jp.array(0, dtype=jp.int32),
          "current_reference_motion": current_reference_motion,
          "imitation_phase": jp.zeros(2, dtype=jp.float32),
      }
  
      # Metrics
      metrics = {}
      for k, v in self._config.reward_config.scales.items():
          if v != 0:
              if v > 0:
                  metrics[f"reward/{k}"] = jp.zeros((), dtype=jp.float32)
              else:
                  metrics[f"cost/{k}"] = jp.zeros((), dtype=jp.float32)
      metrics["swing_peak"] = jp.zeros((), dtype=jp.float32)
  
      # Contacts (ground truth via collision)
      contact = jp.array(
          [geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id],
          dtype=jp.bool_,
      )
  
      # --- Feet kinematics initialization (WORLD frame) ---
      # Use the SAME source as step/obs: sensor-based feet pos/vel.
      nfeet = len(constants.FEET_SITES)
      
      feet_pos_world = jp.stack(
          [data.sensordata[self._foot_pos_sensor_adr[i]] for i in range(nfeet)],
          axis=0,
      ).astype(jp.float32)  # (nfeet, 3)
      
      feet_vel_world = jp.stack(
          [data.sensordata[self._foot_linvel_sensor_adr[i]] for i in range(nfeet)],
          axis=0,
      ).astype(jp.float32)  # (nfeet, 3)
      
      # --- BASE(BODY) frame = floating base body "base" ---
      base_body_id = self._base_body_id
      
      base_pos_world = data.xpos[base_body_id].astype(jp.float32)                         # (3,)
      R_wb = jp.reshape(data.xmat[base_body_id], (3, 3)).astype(jp.float32)               # world-from-base
      R_bw = R_wb.T                                                                       # base-from-world
      
      feet_pos_body = (R_bw @ (feet_pos_world - base_pos_world).T).T                      # (nfeet, 3)
      feet_vel_body = jp.zeros_like(feet_pos_body) 


      # Initialize tracking buffers (WORLD)
      info["prev_feet_pos_world"] = feet_pos_world
      info["current_feet_pos_world"] = feet_pos_world
      info["prev_feet_vel_world"] = feet_vel_world
      info["current_feet_vel_world"] = feet_vel_world

      info["prev_feet_pos_body"] = feet_pos_body
      info["prev_feet_vel_body"] = feet_vel_body
      info["current_feet_pos_body"] = feet_pos_body
      info["current_feet_vel_body"] = feet_vel_body
  
      info["liftoff_feet_world"] = feet_pos_world
      info["liftoff_feet_body"] = feet_pos_body
      info["liftoff_feet_vel_world"] = feet_vel_world
      info["liftoff_feet_vel_body"] = feet_vel_body

      # Base velocities (initialize; will be overwritten in step)
      info["base_linvel_body"]  = jp.zeros((3,), jp.float32)
      info["base_angvel_body"]  = jp.zeros((3,), jp.float32)
      info["base_linvel_world"] = jp.zeros((3,), jp.float32)
      info["base_angvel_world"] = jp.zeros((3,), jp.float32)
  
      nfeet = self._nfeet
  
      info.update({
          # Raw + filtered contact memory
          "last_contact_raw": contact,
          "contact_phase_prev": contact,
  
          # Swing timing
          "feet_air_time": jp.zeros(nfeet, dtype=jp.float32),
          "feet_air_time_prev": jp.zeros(nfeet, dtype=jp.float32),
  
          # Swing height tracking (WORLD z)
          "swing_peak": jp.zeros(nfeet, dtype=jp.float32),
          "swing_peak_prev": jp.zeros(nfeet, dtype=jp.float32),

          "liftoff_speed_body": jp.zeros(nfeet, dtype=jp.float32),
      })
      obs = self._get_obs(data, info, contact)
      reward = jp.array(0.0, dtype=jp.float32)
      done = jp.array(0.0, dtype=jp.float32)
      return mjx_env.State(data, obs, reward, done, metrics, info)
    
  
    # -------------------------
    # Step
    # -------------------------
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
      # Local alias (avoid accidental mixed state.info/info usage)
      info = state.info
  
      # ------------------------------------------------------------
      # Imitation phase + current reference
      # ------------------------------------------------------------
      if USE_IMITATION_REWARD:
          info["imitation_i"] = (info["imitation_i"] + 1) % self.PRM.nb_steps_in_period
          phase = 2.0 * jp.pi * (info["imitation_i"] / jp.asarray(self.PRM.nb_steps_in_period, jp.float32))
          info["imitation_phase"] = jp.array([jp.cos(phase), jp.sin(phase)], dtype=jp.float32)
  
          ref = self.PRM.get_reference_motion(
              info["command"][0], info["command"][1], info["command"][2], info["imitation_i"]
          )
          info["current_reference_motion"] = jp.asarray(ref[: self._ref_dim], dtype=jp.float32)
      else:
          info["imitation_i"] = jp.array(0, dtype=jp.int32)
          info["imitation_phase"] = jp.zeros(2, dtype=jp.float32)
          info["current_reference_motion"] = jp.zeros(0, dtype=jp.float32)
  
      # ------------------------------------------------------------
      # RNG splits
      # ------------------------------------------------------------
      (info["rng"], push1_rng, push2_rng, push_interval_rng, action_delay_rng, hor_rng) = jax.random.split(info["rng"], 6)
  
      # ------------------------------------------------------------
      # Action delay model
      # ------------------------------------------------------------
      info["action_history"] = jp.roll(info["action_history"], self._actuators).at[: self._actuators].set(
          action.astype(jp.float32)
      )
      action_idx = jax.random.randint(
          action_delay_rng,
          (1,),
          minval=self._config.noise_config.action_min_delay,
          maxval=self._config.noise_config.action_max_delay,
      )
      action_w_delay = info["action_history"].reshape((-1, self._actuators))[action_idx[0]]
  
      # ------------------------------------------------------------
      # External push disturbance (no-op if enable=False)
      # ------------------------------------------------------------
      push_theta = jax.random.uniform(push1_rng, (), maxval=2 * jp.pi, dtype=jp.float32)
      push_magnitude = jax.random.uniform(
          push2_rng,
          (),
          minval=self._config.push_config.magnitude_range[0],
          maxval=self._config.push_config.magnitude_range[1],
          dtype=jp.float32,
      )
      push_dir = jp.array([jp.cos(push_theta), jp.sin(push_theta)], dtype=jp.float32)
  
      warmup_steps = jp.asarray(self._config.push_config.warmup_steps, jp.int32)
      after_warmup = info["step"] >= warmup_steps
      interval_hit = (jp.mod(info["push_step"] + 1, info["push_interval_steps"]) == 0)
  
      will_push_now = interval_hit & after_warmup & jp.array(self._config.push_config.enable, dtype=jp.bool_)
      push_mask = will_push_now.astype(jp.float32)
      push = push_dir * push_mask
  
      kick_scale = push_magnitude / jp.maximum(self._torso_mass, 1e-6)
      kick = push * kick_scale
  
      qvel = state.data.qvel
      qvel_xy = qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2]
      qvel = qvel.at[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2].set(qvel_xy + kick)
      state = state.replace(data=state.data.replace(qvel=qvel))
  
      # Resample push interval after a push
      T_min, T_max = self._config.push_config.interval_range
      new_interval_sec = jax.random.uniform(push_interval_rng, (), minval=T_min, maxval=T_max, dtype=jp.float32)
      new_interval_steps = jp.round(new_interval_sec / self.dt).astype(jp.int32)
  
      info["push_interval_steps"] = jp.where(will_push_now, new_interval_steps, info["push_interval_steps"])
      info["push_step"] = jp.where(
          will_push_now,
          jp.array(0, dtype=jp.int32),
          info["push_step"] + 1,
      )
  
      # ------------------------------------------------------------
      # Motor targets + physics step
      # ------------------------------------------------------------
      prev_motor_targets = info["motor_targets"]
      motor_targets = self._compute_motor_targets(prev_motor_targets, action_w_delay)
      data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
      info["motor_targets"] = motor_targets
  
      # ------------------------------------------------------------
      # Kinematics in WORLD + BASE(BODY) frame
      # ------------------------------------------------------------
      nfeet = self._nfeet
      base_body_id = self._base_body_id
      imu_site_id = self._imu_site_id
      
      sd = data.sensordata
      
      # Feet WORLD pos/vel from sensors (already world-frame)
      feet_pos_world = jp.stack([sd[self._foot_pos_sensor_adr[i]] for i in range(nfeet)], axis=0).astype(jp.float32)
      feet_vel_world = jp.stack([sd[self._foot_linvel_sensor_adr[i]] for i in range(nfeet)], axis=0).astype(jp.float32)
      
      # Bookkeeping (WORLD)
      state.info["prev_feet_pos_world"] = state.info["current_feet_pos_world"]
      state.info["prev_feet_vel_world"] = state.info["current_feet_vel_world"]
      state.info["current_feet_pos_world"] = feet_pos_world
      state.info["current_feet_vel_world"] = feet_vel_world
      
      # -------------------------
      # BASE pose/orientation
      # -------------------------
      base_pos_world = data.xpos[base_body_id].astype(jp.float32)                       # (3,)
      R_wb = jp.reshape(data.xmat[base_body_id], (3, 3)).astype(jp.float32)             # world-from-base
      R_bw = R_wb.T                                                                     # base-from-world
      
      # -------------------------
      # IMU pose/orientation (for offset correction + frame mapping)
      # -------------------------
      imu_pos_world = data.site_xpos[imu_site_id].astype(jp.float32)                    # (3,)
      R_wi = jp.reshape(data.site_xmat[imu_site_id], (3, 3)).astype(jp.float32)         # world-from-imu
      r_w = (imu_pos_world - base_pos_world).astype(jp.float32)                         # (3,) base->imu in WORLD
      
      # IMU -> BASE rotation (base-from-imu)
      R_bi = (R_bw @ R_wi).astype(jp.float32)                                           # (3,3)
      
      # -------------------------
      # Angular velocity
      # -------------------------
      # Gyro is expressed in IMU frame. Rotate into BASE frame explicitly.
      imu_angvel_i = jp.nan_to_num(self.get_gyro(data)).astype(jp.float32)              # (3,) IMU frame
      base_angvel_body = (R_bi @ imu_angvel_i).astype(jp.float32)                       # (3,) BASE frame
      
      # World angular velocity (from BASE frame)
      base_angvel_world = (R_wb @ base_angvel_body).astype(jp.float32)                  # (3,) world
      
      # -------------------------
      # Linear velocity (base origin, expressed in base frame)
      # -------------------------
      # local_linvel is IMU-point linear velocity expressed in IMU frame.
      imu_linvel_i = jp.nan_to_num(self.get_local_linvel(data)).astype(jp.float32)      # (3,) IMU frame
      
      # Convert IMU-point linvel to WORLD using IMU orientation (not base)
      imu_linvel_world = (R_wi @ imu_linvel_i).astype(jp.float32)                       # (3,) world (IMU point)
      
      # Correct IMU-point velocity -> base-origin velocity in WORLD:
      # v_base = v_imu - ω_world × (p_imu - p_base)
      base_linvel_world = (imu_linvel_world - jp.cross(base_angvel_world, r_w)).astype(jp.float32)
      
      # Express base linear velocity in BASE frame
      base_linvel_body = (R_bw @ base_linvel_world).astype(jp.float32)                  # (3,) base frame
      
      # Keep same info keys/variable names as before
      state.info["base_linvel_body"] = base_linvel_body
      state.info["base_angvel_body"] = base_angvel_body
      state.info["base_linvel_world"] = base_linvel_world
      state.info["base_angvel_world"] = base_angvel_world
      
      # -------------------------
      # Feet in BASE(BODY) frame (origin = base origin, axes = base axes)
      # -------------------------
      feet_pos_body = (R_bw @ (feet_pos_world - base_pos_world).T).T                    # (nfeet,3)
      
      v_rel_b_linear = (R_bw @ (feet_vel_world - base_linvel_world).T).T                # (nfeet,3)
      feet_vel_body = v_rel_b_linear - jp.cross(
          jp.broadcast_to(base_angvel_body, v_rel_b_linear.shape),
          feet_pos_body,
      )
      
      # Bookkeeping (BODY)
      state.info["prev_feet_pos_body"] = state.info["current_feet_pos_body"]
      state.info["prev_feet_vel_body"] = state.info["current_feet_vel_body"]
      state.info["current_feet_pos_body"] = jp.nan_to_num(feet_pos_body).astype(jp.float32)
      state.info["current_feet_vel_body"] = jp.nan_to_num(feet_vel_body).astype(jp.float32)
      

      # ------------------------------------------------------------
      # Contacts + events + airtime + swing peak + liftoff snapshots
      # ------------------------------------------------------------
      # --- Raw geometric contact (world) ---
      contact_raw = jp.array(
          [geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id],
          dtype=jp.bool_,
      )  # (nfeet,)
      
      # --- 1-step hysteresis contact for gait phase ---
      contact_phase = contact_raw | info["last_contact_raw"]  # (nfeet,)
      
      # Cache previous buffers BEFORE modifying them (for touchdown rewards)
      feet_air_time_prev = info["feet_air_time"]   # (nfeet,)
      swing_peak_prev = info["swing_peak"]         # (nfeet,)
      
      # --- Events computed on filtered contact (prev -> current) ---
      touchdown = (~info["contact_phase_prev"]) & contact_phase  # (nfeet,)
      liftoff   = info["contact_phase_prev"] & (~contact_phase)  # (nfeet,)
      liftoff_raw   = info["last_contact_raw"] & (~contact_raw)

      # --- first_contact: touchdown gated by *previous* airtime ---
      # (use prev airtime; do NOT use values after resetting)
      first_contact = (touchdown & (feet_air_time_prev > jp.asarray(0.05, jp.float32))).astype(jp.float32)
      
      # --- Update airtime: accumulate only in swing, reset in contact ---
      info["feet_air_time"] = feet_air_time_prev + jp.asarray(self.dt, jp.float32) * (~contact_phase).astype(jp.float32)
      info["feet_air_time"] = jp.where(contact_phase, jp.zeros_like(info["feet_air_time"]), info["feet_air_time"])
      
      # --- Swing peak z tracking (WORLD), accumulate only in swing, reset on contact ---
      info["swing_peak"] = jp.maximum(swing_peak_prev, feet_pos_world[:, 2])
      info["swing_peak"] = jp.where(contact_phase, jp.zeros_like(info["swing_peak"]), info["swing_peak"])
      
      # Store touchdown-moment values so rewards can use them
      info["feet_air_time_prev"] = feet_air_time_prev
      info["swing_peak_prev"] = swing_peak_prev
      
      # --- Liftoff snapshots (store at liftoff event) ---
      info["liftoff_feet_world"] = jp.where(liftoff_raw[:, None], feet_pos_world, info["liftoff_feet_world"])
      info["liftoff_feet_vel_world"] = jp.where(liftoff_raw[:, None], feet_vel_world, info["liftoff_feet_vel_world"])
      info["liftoff_feet_body"] = jp.where(liftoff_raw[:, None], info["current_feet_pos_body"], info["liftoff_feet_body"])
      info["liftoff_feet_vel_body"] = jp.where(liftoff_raw[:, None], info["current_feet_vel_body"], info["liftoff_feet_vel_body"])
      
      liftoff_speed_body_now = jp.linalg.norm(info["current_feet_vel_body"][:, :2], axis=-1)
      info["liftoff_speed_body"] = jp.where(liftoff_raw, liftoff_speed_body_now, info["liftoff_speed_body"])
      
      # --- Update contact memories (shape-stable) ---
      info["contact_phase_prev"] = contact_phase
      info["last_contact_raw"] = contact_raw
      
  
      # ------------------------------------------------------------
      # Command horizon logic
      # ------------------------------------------------------------
      info["cmd_step"] = info["cmd_step"] + 1
      resample = info["cmd_step"] >= info["cmd_horizon_steps"]
  
      new_cmd_horizon_steps = self._sample_cmd_horizon_steps(hor_rng)
      info["rng"], cmd_rng = jax.random.split(info["rng"])
      new_cmd = self.sample_command(cmd_rng)
  
      info["command"] = jp.where(resample, new_cmd, info["command"])
      info["cmd_step"] = jp.where(resample, jp.array(0, dtype=jp.int32), info["cmd_step"])
      info["cmd_horizon_steps"] = jp.where(resample, new_cmd_horizon_steps, info["cmd_horizon_steps"])
  
      # ------------------------------------------------------------
      # Obs / termination
      # ------------------------------------------------------------
      obs = self._get_obs(data, info, contact_phase)
      done = self._get_termination(data)
  
      # ------------------------------------------------------------
      # Rewards
      # ------------------------------------------------------------
      rewards = self._get_reward(data, action, info, state.metrics, done, first_contact, contact_phase, contact_raw)
  
      scaled_rewards = {}
      for k, v in rewards.items():
          scale = jp.asarray(self._config.reward_config.scales[k], jp.float32)
          scaled_rewards[k] = v * scale
  
      reward = sum(scaled_rewards.values())
      reward = jp.clip(reward, -10000.0, 10000.0)
  
      # ------------------------------------------------------------
      # Bookkeeping
      # ------------------------------------------------------------
      info["push"] = push
      info["step"] = info["step"] + 1
  
      info["last_last_last_act"] = info["last_last_act"]
      info["last_last_act"] = info["last_act"]
      info["last_act"] = action.astype(jp.float32)
  
      for k, v in scaled_rewards.items():
          rew_scale = self._config.reward_config.scales[k]
          if rew_scale != 0:
              if rew_scale > 0:
                  state.metrics[f"reward/{k}"] = v
              else:
                  state.metrics[f"cost/{k}"] = -v
  
      state.metrics["swing_peak"] = jp.mean(info["swing_peak"])
  
      done_f = done.astype(jp.float32)
      state = state.replace(data=data, obs=obs, reward=reward, done=done_f, info=info)
      return state
  

  # -------------------------
  # Termination
  # -------------------------
  
  def _get_termination(self, data: mjx.Data) -> jax.Array:
      R_wb = jp.reshape(data.xmat[self._torso_body_id], (3, 3))
      z_body_world = R_wb[:, 2]
      cos_tilt = z_body_world[2]
  
      tilt_limit_rad = jp.deg2rad(jp.asarray(self._config.term_max_tilt_deg, jp.float32))
      cos_limit = jp.cos(tilt_limit_rad)
      too_tilted = cos_tilt < cos_limit
  
      # Height termination based on torso/IMU site height
      torso_z = data.site_xpos[self._site_id, 2]
      too_low = torso_z < jp.asarray(self._config.term_height_min, jp.float32)
  
      bad_state = (~jp.isfinite(data.qpos).all()) | (~jp.isfinite(data.qvel).all())
  
      return too_tilted | too_low | bad_state
  
  
  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
      # -------------------------
      # BODY frame = floating base ("base")
      # -------------------------
      base_body_id = self._base_body_id
      R_wb = jp.reshape(data.xmat[base_body_id], (3, 3)).astype(jp.float32)   # world-from-base
      R_bw = R_wb.T                                                          # base-from-world
  
      # -------------------------
      # IMU (noisy for policy)
      # -------------------------
      gyro = self.get_gyro(data)  # IMU/site frame; in your model IMU axes ~= base axes
      info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_gyro = (
          gyro
          + (2.0 * jax.random.uniform(noise_rng, shape=gyro.shape, dtype=jp.float32) - 1.0)
          * self._config.noise_config.level
          * self._config.noise_config.scales.gyro
      )
  
      accelerometer = self.get_accelerometer(data)  # IMU/site frame; in your model IMU axes ~= base axes
      info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_accelerometer = (
          accelerometer
          + (2.0 * jax.random.uniform(noise_rng, shape=accelerometer.shape, dtype=jp.float32) - 1.0)
          * self._config.noise_config.level
          * self._config.noise_config.scales.accelerometer
      )
  
      # -------------------------
      # Joints (no backlash)
      # -------------------------
      joint_angles = self.get_actuator_joints_qpos(data.qpos)
      info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_joint_angles = (
          joint_angles
          + (2.0 * jax.random.uniform(noise_rng, shape=joint_angles.shape, dtype=jp.float32) - 1.0)
          * self._config.noise_config.level
          * self._qpos_noise_scale
      )
  
      joint_vel = self.get_actuator_joints_qvel(data.qvel)
      info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_joint_vel = (
          joint_vel
          + (2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape, dtype=jp.float32) - 1.0)
          * self._config.noise_config.level
          * self._config.noise_config.scales.joint_vel
      )
  
      # -------------------------
      # Optional (kept for privileged; body-frame below is consistent)
      # -------------------------
      linvel_local = self.get_local_linvel(data)  # IMU/site frame; not necessarily available on hardware
  
      # -------------------------
      # NaN-safe policy signals
      # -------------------------
      noisy_gyro = jp.nan_to_num(noisy_gyro)
      noisy_accelerometer = jp.nan_to_num(noisy_accelerometer)
      noisy_joint_angles = jp.nan_to_num(noisy_joint_angles)
      noisy_joint_vel = jp.nan_to_num(noisy_joint_vel)
      linvel_local = jp.nan_to_num(linvel_local)
  
      # -------------------------
      # Policy state (NO GRAVITY)
      # -------------------------
      state = jp.hstack(
          [
              noisy_gyro,                                              # 3
              noisy_accelerometer,                                     # 3
              info["command"],                                         # 3
              noisy_joint_angles - self._default_actuator,             # nu
              noisy_joint_vel * self._config.dof_vel_scale,            # nu
              info["last_act"],                                        # nu
              info["last_last_act"],                                   # nu
              info["last_last_last_act"],                              # nu
              info["motor_targets"],                                   # nu
              info["imitation_phase"],                                 # 2
          ]
      )
  
      # -------------------------
      # Privileged signals (clean/no-noise) - BODY FRAME CONSISTENT
      # -------------------------
      gyro_clean = jp.nan_to_num(gyro)
      accelerometer_clean = jp.nan_to_num(accelerometer)
  
      # "Clean gravity" expressed in BASE(BODY) frame (not IMU site frame)
      gravity_body_clean = jp.nan_to_num(R_bw @ jp.array([0.0, 0.0, -1.0], dtype=jp.float32))
  
      # Base/world velocities: prefer what step() already computed (now base-frame consistent)
      base_linvel_world = jp.nan_to_num(info["base_linvel_world"])
      base_angvel_world = jp.nan_to_num(info["base_angvel_world"])
      base_linvel_body  = jp.nan_to_num(info["base_linvel_body"])
      base_angvel_body  = jp.nan_to_num(info["base_angvel_body"])
  
      # If you still want these MuJoCo sensor readouts for debugging, keep them,
      # but note they are IMU-site quantities (not base-origin). Use with care.
      global_angvel_site_world = jp.nan_to_num(self.get_global_angvel(data))
      linvel_local_clean = jp.nan_to_num(linvel_local)
  
      joint_angles_clean = jp.nan_to_num(joint_angles - self._default_actuator)
      joint_vel_clean = jp.nan_to_num(joint_vel)
  
      # Height: use BASE body height (consistent), not IMU site height
      base_z = jp.nan_to_num(
          data.xpos[base_body_id, 2],
          nan=self._config.base_height_nominal,
          posinf=self._config.base_height_nominal,
          neginf=self._config.base_height_nominal,
      )
  
      actuator_force = jp.nan_to_num(data.actuator_force)
  
      # Feet signals: take from step() bookkeeping (already base-frame consistent)
      feet_pos_world = jp.nan_to_num(info["current_feet_pos_world"]).ravel()
      feet_vel_world = jp.nan_to_num(info["current_feet_vel_world"]).ravel()
      feet_pos_body  = jp.nan_to_num(info["current_feet_pos_body"]).ravel()
      feet_vel_body  = jp.nan_to_num(info["current_feet_vel_body"]).ravel()
  
      liftoff_pos_world = jp.nan_to_num(info["liftoff_feet_world"]).ravel()
      liftoff_pos_body  = jp.nan_to_num(info["liftoff_feet_body"]).ravel()
      liftoff_vel_world = jp.nan_to_num(info["liftoff_feet_vel_world"]).ravel()
      liftoff_vel_body  = jp.nan_to_num(info["liftoff_feet_vel_body"]).ravel()
  
      privileged_state = jp.hstack(
          [
              state,  # include policy state
  
              # Clean proprioception/IMU
              gyro_clean,
              accelerometer_clean,
              gravity_body_clean,
  
              # Clean base twist (BODY frame) + optional world
              base_linvel_body,
              base_angvel_body,
              base_linvel_world,
              base_angvel_world,
  
              # Optional debug-only (IMU-site/world)
              linvel_local_clean,
              global_angvel_site_world,
  
              # Joints / actuation
              joint_angles_clean,
              joint_vel_clean,
              base_z[None],
              actuator_force,
  
              # Contacts
              contact.astype(jp.float32),
  
              # Feet kinematics (BODY frame + liftoff)
              feet_pos_body,
              feet_vel_body,
              liftoff_pos_body,
              liftoff_vel_body,
  
              # Gait / imitation metadata
              info["feet_air_time"],
              info["current_reference_motion"],
              info["imitation_i"][None].astype(jp.float32),
              info["imitation_phase"],
          ]
      )
  
      return {"state": state, "privileged_state": privileged_state}
  
  
  
    # -------------------------
    # Reward
    # -------------------------
  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,   # (nfeet,) float/bool touchdown event
      contact: jax.Array,
      raw_contact: jax.Array,         # (nfeet,) bool contact (use the SAME signal used in step transitions)
  ) -> dict[str, jax.Array]:
      del metrics  # Unused.
  
      cfg = self._config.reward_config
  
      # -------------------------
      # Pull commonly used signals from info (prefer info over raw sensordata)
      # -------------------------
      cmd = info["command"]
      cmd_speed = jp.linalg.norm(cmd[:2])
      yaw_cmd = cmd[2]
      
      # Base velocities (BODY frame, single source)
      linvel_body = jp.nan_to_num(info["base_linvel_body"]).astype(jp.float32)
      angvel_body = jp.nan_to_num(info["base_angvel_body"]).astype(jp.float32)
      
      # Feet kinematics (WORLD + BODY) (single source)
      cur_feet_world  = jp.nan_to_num(info["current_feet_pos_world"]).astype(jp.float32)
      prev_feet_world = jp.nan_to_num(info["prev_feet_pos_world"]).astype(jp.float32)
      
      cur_feet_body   = jp.nan_to_num(info["current_feet_pos_body"]).astype(jp.float32)
      prev_feet_body  = jp.nan_to_num(info["prev_feet_pos_body"]).astype(jp.float32)
      
      feet_vel_world  = jp.nan_to_num(info["current_feet_vel_world"]).astype(jp.float32)
      feet_vel_body   = jp.nan_to_num(info["current_feet_vel_body"]).astype(jp.float32)
      
      liftoff_feet_body = jp.nan_to_num(info["liftoff_feet_body"]).astype(jp.float32)
      liftoff_feet_vel_body = jp.nan_to_num(info["liftoff_feet_vel_body"]).astype(jp.float32)
  
      # Contact as float for some reward helpers
      contact_f = contact.astype(jp.float32)
      first_contact_f = first_contact.astype(jp.float32)
      contact_r = raw_contact.astype(jp.float32)
  
      # Feet speed in XY for stillness (use BODY or WORLD; BODY is yaw-invariant)
      feet_speed_xy = jp.nan_to_num(jp.linalg.norm(feet_vel_body[:, :2], axis=-1).sum())
  
      # -------------------------
      # Base rewards / costs (UNSCALED)
      # -------------------------
      ret = {
          "tracking_lin_vel": reward_tracking_lin_vel(
              cmd, jp.nan_to_num(self.get_local_linvel(data)).astype(jp.float32), cfg.tracking_sigma
          ),
          "tracking_ang_vel": reward_tracking_ang_vel(
              cmd, angvel_body, cfg.tracking_sigma
          ),
          "torques": cost_torques(jp.nan_to_num(data.actuator_force)),
          "action_rate": cost_action_rate(
              action.astype(jp.float32), info["last_act"]
          ),
          "alive": reward_alive(),
      }
  
      # -------------------------
      # Imitation (use BODY-frame feet for consistency with PRM, and BODY-frame base vels)
      # -------------------------
      # NOTE: reward_imitation signature in your code expects:
      #   (global_linvel, global_angvel, joint_qpos, joint_qvel, ee_pos, ee_vel, contact, ref, cmd, enabled)
      # We now pass:
      #   - base_linvel_world/base_angvel_world if you want "world", but we keep existing API calls.
      #   - ee_pos/ee_vel from BODY frame, since your ref motion now uses root/base frame for toes.
      #
      # If your imitation code still expects WORLD-frame feet, swap to cur_feet_world/feet_vel_world.
      imi, dbg = reward_imitation(
          linvel_body,
          angvel_body,
          jp.nan_to_num(self.get_actuator_joints_qpos(data.qpos)),
          jp.nan_to_num(self.get_actuator_joints_qvel(data.qvel)),
          jp.nan_to_num(cur_feet_body),    
          jp.nan_to_num(feet_vel_body),   
          contact_r,
          info["current_reference_motion"],
          cmd,
          USE_IMITATION_REWARD,
      )
      ret["imitation"] = imi
  
      log = (info["step"] % 1) == 0
    #   jdebug.print(
    #       "imi={imi:.3f} pose={p:.3f} jvel={jv:.3f} root={r:.3f} eeP={ep:.3f} eeV={ev:.3f} c={c:.3f}",
    #       imi=jp.where(log, imi, 0.0),
    #       p=jp.where(log, dbg["r_pose"], 0.0),
    #       jv=jp.where(log, dbg["r_jvel"], 0.0),
    #       r=jp.where(log, dbg["r_root"], 0.0),
    #       ep=jp.where(log, dbg["r_ee_pos"], 0.0),
    #       ev=jp.where(log, dbg["r_ee_vel"], 0.0),
    #       c=jp.where(log, dbg["r_contact"], 0.0),
    #   )
  
      # -------------------------
      # Shaping: orientation, feet behavior, height
      # -------------------------
      ret["upright"] = self.reward_upright(data)
  
      # Foot slip: use BODY-frame feet velocity (yaw-invariant), no sensordata indexing here
      ret["foot_slip"] = self.cost_foot_slip(feet_vel_body, contact_phase=contact)
  
      # Foot clearance at touchdown: uses swing peak (world z), paid only on touchdown
      ret["foot_clearance"] = self.reward_foot_clearance_touchdown(
          swing_peak_z=info["swing_peak_prev"],
          first_contact=first_contact_f,
          command=cmd,
          z_target=cfg.foot_clearance_target,  # e.g. 0.04
      )
  
      # Feet air time at touchdown: uses feet_air_time BEFORE reset + swing_peak gates
      ret["feet_air_time"] = self.reward_feet_air_time_touchdown(
          feet_air_time=info["feet_air_time_prev"],
          first_contact=first_contact_f,
          swing_peak_z=info["swing_peak_prev"],
          command=cmd,
          target_air_time=cfg.feet_air_time_threshold,  # e.g. 0.40
      )
  
      # Knee flexion (dense when moving)
      joint_angles = jp.nan_to_num(self.get_actuator_joints_qpos(data.qpos))
      ret["knee_flexion"] = self.reward_knee_flexion(joint_angles, cmd)
  
      # Step distance (touchdown-based, BODY frame, uses liftoff snapshots)
      ret["step_distance"] = self.reward_step_distance_touchdown(
          liftoff_feet_body=info["liftoff_feet_body"],
          curr_feet_body=info["current_feet_pos_body"],
          command=info["command"],
          first_contact=first_contact,
          feet_air_time=info["feet_air_time"],
          swing_peak_z=info["swing_peak"],
      )
      ret["swing_progress"] = self.reward_swing_progress_dense(
          feet_vel_body=feet_vel_body,
          contact_phase=contact,
          command=cmd,
      )
  
      # Foot separation (BODY frame)
      ret["foot_seperation"] = self.reward_foot_separation(
          feet_pos_body=cur_feet_body,
          command=cmd,
          target=0.16,
      )
      ret["foot_flat"] = self.cost_foot_flat_contact(
          data,
          contact_phase=contact_f,     
          feet_site_id=self._feet_site_id,
          use_exp=False,                 
      )
      
      # Laziness cost: dense base speed + touchdown step shortfall
      ret["lazyness"] = self.cost_lazy(
          liftoff_feet_body=liftoff_feet_body,
          curr_feet_body=cur_feet_body,
          first_contact=first_contact_f,
          cmd_speed=cmd_speed,
      )
  
      # -------------------------
      # Stillness + yaw drift when command is ~0
      # -------------------------
      still_mask = (cmd_speed < 0.01) & (jp.abs(yaw_cmd) < 0.01)
      stillness_pen = jp.where(
          still_mask,
          jp.linalg.norm(linvel_body[:2]) + feet_speed_xy,
          jp.array(0.0, dtype=jp.float32),
      )
      ret["stillness"] = jp.nan_to_num(stillness_pen)
  
      # Yaw drift cost when yaw command is ~0
      yaw_rate = angvel_body[2]
      yaw_drift_pen = jp.where(
          jp.abs(yaw_cmd) < 1e-3,
          yaw_rate**2,
          jp.array(0.0, dtype=jp.float32),
      )
      ret["yaw_drift"] = jp.nan_to_num(yaw_drift_pen)
  
      # Termination penalty (1 on done, 0 otherwise; scaled by negative factor)
      ret["termination"] = done.astype(jp.float32)
  
      # -------------------------
      # Make every component finite & clip extreme magnitudes
      # -------------------------
      CLIP_MAG = jp.asarray(1e3, jp.float32)  # per-term limit before scaling
      for k, v in ret.items():
          v = jp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
          ret[k] = jp.clip(v, -CLIP_MAG, CLIP_MAG)
  
      # -------------------------
      # Debug print: show scaled total + a few key components
      # -------------------------
      def _compute_total_reward(ret_dict, scales_cfg):
          total = jp.array(0.0, dtype=jp.float32)
          for name, val in ret_dict.items():
              scale = jp.asarray(scales_cfg.scales[name], jp.float32)
              total = total + val * scale
          return total
  
      total_reward_est = _compute_total_reward(ret, cfg)
  
      sc = cfg.scales
      log = (info["step"] % 1) == 0
    #   jdebug.print(
    #      "td={td} fc={fc} air={air} peak={peak} cmd={cmd}",
    #      td=jp.where(log, jp.sum(touchdown.astype(jp.int32)), 0),
    #      fc=jp.where(log, jp.sum(first_contact.astype(jp.int32)), 0),
    #      air=jp.where(log, feet_air_time_prev, 0.0),
    #      peak=jp.where(log, swing_peak_prev, 0.0),
    #      cmd=jp.where(log, cmd_speed, 0.0),
    #  )

    #   R_wb = jp.reshape(data.xmat[self._base_body_id], (3, 3))  # world-from-base
    #   base_x_world = R_wb[:, 0]
    #   base_y_world = R_wb[:, 1]
    #   base_z_world = R_wb[:, 2]
    #   wx = jp.array([1., 0., 0.], jp.float32)
    #   wy = jp.array([0., 1., 0.], jp.float32)
    #   wz = jp.array([0., 0., 1.], jp.float32)
      
    #   jax.debug.print(
    #       "base_x·[wx,wy,wz]=[{:.3f},{:.3f},{:.3f}]  base_y·[wx,wy,wz]=[{:.3f},{:.3f},{:.3f}]  base_z·wz={:.3f}",
    #       jp.dot(base_x_world, wx), jp.dot(base_x_world, wy), jp.dot(base_x_world, wz),
    #       jp.dot(base_y_world, wx), jp.dot(base_y_world, wy), jp.dot(base_y_world, wz),
    #       jp.dot(base_z_world, wz),
    #   )


  
    #   jdebug.print(
    #       "step={step} tot={tot:.1f} imi={imi:.2f} vlin={vlin:.2f} vang={vang:.2f} stepd={stepd:.2f} slip={slip:.2f} clear={clear:.2f} airt={airt:.2f} knee={knee:.2f} still={still:.2f}",
    #       step=info["step"],
    #       tot=jp.where(log, total_reward_est, 0.0),
    #       imi=jp.where(log, ret["imitation"] * sc.imitation, 0.0),
    #       vlin=jp.where(log, ret["tracking_lin_vel"] * sc.tracking_lin_vel, 0.0),
    #       vang=jp.where(log, ret["tracking_ang_vel"] * sc.tracking_ang_vel, 0.0),
    #       stepd=jp.where(log, ret["step_distance"] * sc.step_distance, 0.0),
    #       slip=jp.where(log, ret["foot_slip"] * sc.foot_slip, 0.0),
    #       clear=jp.where(log, ret["foot_clearance"] * sc.foot_clearance, 0.0),
    #       airt=jp.where(log, ret["feet_air_time"] * sc.feet_air_time, 0.0),
    #       knee=jp.where(log, ret["knee_flexion"] * sc.knee_flexion, 0.0),
    #       still=jp.where(log, ret["stillness"] * sc.stillness, 0.0),
    #   )
  
      return ret
  
  
  
    # -------------------------
    # Command sampling
    # -------------------------
  
  def sample_command(self, rng: jax.Array) -> jax.Array:
      # Only linear x/y velocity and yaw rate (no head/neck).
      rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

      lin_vel_x = jax.random.choice(
          rng1,
          jp.array(self._config.lin_vel_x, dtype=jp.float32)
      )

      lin_vel_y = jax.random.choice(
          rng2,
          jp.array(self._config.lin_vel_y, dtype=jp.float32)
      )

      ang_vel_yaw = jax.random.choice(
          rng3,
         jp.array(self._config.ang_vel_yaw, dtype=jp.float32)
      )

      cmd = jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw])
      hold_still = jax.random.bernoulli(rng4, p=0.15)
      return jp.where(hold_still, jp.zeros(3, dtype=jp.float32), cmd)
