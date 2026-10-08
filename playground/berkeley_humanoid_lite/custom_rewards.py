import jax
import jax.numpy as jp

def reward_imitationnew(
    base_qpos: jax.Array,
    base_qvel: jax.Array,
    joints_qpos: jax.Array,
    joints_qvel: jax.Array,
    contacts: jax.Array,
    reference_frame: jax.Array,
    cmd: jax.Array,
    use_imitation_reward: bool = False,
) -> jax.Array:
    """Imitation reward for BHL Joystick task (no head/neck).

    Assumed layout of `reference_frame`:
        [ joint_pos (nu),
          joint_vel (nu),
          foot_contacts (nfeet),
          base_lin_vel (3),
          base_ang_vel (3) ]

    Returns a positive, bounded scalar: higher = better imitation.
    """

    # Off-switch or missing reference → zero reward.
    if (not use_imitation_reward) or (reference_frame.size == 0):
        return jp.array(0.0, dtype=jp.float32)

    # ------------------------------------------------------------------
    # Decode reference_frame layout
    # ------------------------------------------------------------------
    n_joints = joints_qpos.shape[0]
    n_feet = contacts.shape[0]

    idx = 0
    ref_joint_pos = reference_frame[idx : idx + n_joints]; idx += n_joints
    ref_joint_vel = reference_frame[idx : idx + n_joints]; idx += n_joints
    ref_foot_contacts = reference_frame[idx : idx + n_feet]; idx += n_feet
    ref_base_lin_vel = reference_frame[idx : idx + 3];       idx += 3
    ref_base_ang_vel = reference_frame[idx : idx + 3]

    # ------------------------------------------------------------------
    # Current sim quantities
    # ------------------------------------------------------------------
    # base_qpos is unused here, but kept in the signature for flexibility.
    base_lin_vel = base_qvel[:3]   # [vx, vy, vz]
    base_ang_vel = base_qvel[3:6]  # [wx, wy, wz]

    joint_pos = joints_qpos
    joint_vel = joints_qvel

    # Command info
    cmd_xy = cmd[:2]
    cmd_speed = jp.linalg.norm(cmd_xy)
    moving = cmd_speed > 0.1

    # ------------------------------------------------------------------
    # 1) Command-consistent speed (directional + sideways penalty)
    # ------------------------------------------------------------------
    v_xy = base_lin_vel[:2]

    eps = jp.asarray(1e-6, dtype=jp.float32)
    default_dir = jp.array([1.0, 0.0], dtype=jp.float32)  # arbitrary when not moving
    cmd_dir = jp.where(
        moving,
        cmd_xy / (cmd_speed + eps),
        default_dir,
    )

    # Parallel / perpendicular components of velocity w.r.t. command direction.
    v_par = jp.dot(v_xy, cmd_dir)
    v_perp = jp.dot(v_xy, jp.array([-cmd_dir[1], cmd_dir[0]], dtype=jp.float32))

    # Match speed along command direction.
    sigma_speed = jp.asarray(0.3, dtype=jp.float32)  # ~0.3 m/s tolerance
    speed_err = v_par - cmd_speed
    speed_along_rew = jp.exp(- (speed_err ** 2) / (2.0 * sigma_speed ** 2))

    # Penalize sideways velocity (perpendicular).
    sigma_side = jp.asarray(0.2, dtype=jp.float32)
    side_pen = jp.exp(- (v_perp ** 2) / (2.0 * sigma_side ** 2))

    # Combine; neutral (1.0) if not moving.
    speed_rew = jp.where(
        moving,
        speed_along_rew * side_pen,
        jp.array(1.0, dtype=jp.float32),
    )

    # ------------------------------------------------------------------
    # 2) Base velocity vs reference (linear + angular)
    # ------------------------------------------------------------------
    lin_vel_err = jp.mean(jp.square(base_lin_vel - ref_base_lin_vel))
    sigma_base_lin = jp.asarray(0.5, dtype=jp.float32)
    base_lin_rew = jp.exp(- lin_vel_err / (2.0 * sigma_base_lin ** 2))

    ang_vel_err = jp.mean(jp.square(base_ang_vel - ref_base_ang_vel))
    sigma_base_ang = jp.asarray(1.0, dtype=jp.float32)
    base_ang_rew = jp.exp(- ang_vel_err / (2.0 * sigma_base_ang ** 2))

    base_vel_rew = 0.5 * base_lin_rew + 0.5 * base_ang_rew

    # ------------------------------------------------------------------
    # 3) Joint tracking: pose + velocity  → [0,1] scores
    # ------------------------------------------------------------------
    joint_pos_err = jp.mean(jp.square(joint_pos - ref_joint_pos))
    joint_vel_err = jp.mean(jp.square(joint_vel - ref_joint_vel))

    sigma_pos = jp.asarray(0.35, dtype=jp.float32)  # ~20 degrees
    sigma_vel = jp.asarray(2.0, dtype=jp.float32)   # ~2 rad/s

    joint_pos_rew = jp.exp(- joint_pos_err / (2.0 * sigma_pos ** 2))
    joint_vel_rew = jp.exp(- joint_vel_err / (2.0 * sigma_vel ** 2))

    # ------------------------------------------------------------------
    # 4) Foot contact pattern match → fraction in [0,1]
    # ------------------------------------------------------------------
    ref_foot_contacts_bin = jp.where(
        ref_foot_contacts > 0.5,
        jp.ones_like(ref_foot_contacts),
        jp.zeros_like(ref_foot_contacts),
    )

    contact_match = (contacts == ref_foot_contacts_bin).astype(jp.float32)
    contact_rew = jp.mean(contact_match)  # 0 = all wrong, 1 = all match

    # ------------------------------------------------------------------
    # 5) Combine with positive weights
    # ------------------------------------------------------------------
    # These are internal weights; env-level imitation scale amplifies the whole thing.
    w_joint_pos = jp.asarray(3.0, dtype=jp.float32)
    w_joint_vel = jp.asarray(1.0, dtype=jp.float32)
    w_speed     = jp.asarray(4.0, dtype=jp.float32)
    w_base      = jp.asarray(2.0, dtype=jp.float32)
    w_contact   = jp.asarray(1.0, dtype=jp.float32)

    reward = (
        w_joint_pos * joint_pos_rew +
        w_joint_vel * joint_vel_rew +
        w_speed     * speed_rew     +
        moving.astype(jp.float32) * (
            w_base    * base_vel_rew +
            w_contact * contact_rew
        )
    )

    # ------------------------------------------------------------------
    # 6) Make it NaN-safe & softly bounded
    # ------------------------------------------------------------------
    reward = jp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    # Max possible is roughly sum of weights; clip a bit above that.
    reward = jp.clip(reward, 0.0, 12.0)

    return reward.astype(jp.float32)


def reward_imitation(
    base_qpos: jax.Array,
    base_qvel: jax.Array,
    joints_qpos: jax.Array,
    joints_qvel: jax.Array,
    contacts: jax.Array,
    reference_frame: jax.Array,
    cmd: jax.Array,
    use_imitation_reward: bool = False,
) -> jax.Array:
    if not use_imitation_reward:
        return jp.nan_to_num(0.0)

    # TODO don't reward for moving when the command is zero.
    cmd_norm = jp.linalg.norm(cmd[:3])

    w_torso_pos = 1.0
    w_torso_orientation = 1.0
    w_lin_vel_xy = 1.0
    w_lin_vel_z = 1.0
    w_ang_vel_xy = 0.5
    w_ang_vel_z = 0.5
    w_joint_pos = 15.0
    w_joint_vel = 1.0e-3
    w_contact = 1.0

    #  TODO : double check if the slices are correct
    linear_vel_slice_start = 26
    linear_vel_slice_end =29    
    angular_vel_slice_start = 29
    angular_vel_slice_end = 32 
    
    joint_pos_slice_start = 0
    joint_pos_slice_end = 12

    joint_vels_slice_start = 12
    joint_vels_slice_end = 24

    # root_pos_slice_start = 0
    # root_pos_slice_end = 3

    root_quat_slice_start = 3
    root_quat_slice_end = 7

    # left_toe_pos_slice_start = 23
    # left_toe_pos_slice_end = 26

    # right_toe_pos_slice_start = 26
    # right_toe_pos_slice_end = 29

    foot_contacts_slice_start = 24
    foot_contacts_slice_end = 26

    # ref_base_pos = reference_frame[root_pos_slice_start:root_pos_slice_end]
    # base_pos = qpos[:3]

    ref_base_orientation_quat = reference_frame[
        root_quat_slice_start:root_quat_slice_end
    ]
    ref_base_orientation_quat = ref_base_orientation_quat / jp.linalg.norm(
        ref_base_orientation_quat
    )  # normalize the quat
    base_orientation = base_qpos[3:7]
    base_orientation = base_orientation / jp.linalg.norm(
        base_orientation
    )  # normalize the quat

    ref_base_lin_vel = reference_frame[linear_vel_slice_start:linear_vel_slice_end]
    base_lin_vel = base_qvel[:3]

    ref_base_ang_vel = reference_frame[angular_vel_slice_start:angular_vel_slice_end]
    base_ang_vel = base_qvel[3:6]

    ref_joint_pos = reference_frame[joint_pos_slice_start:joint_pos_slice_end]
    # remove neck head and antennas
    ref_joint_pos = jp.concatenate([ref_joint_pos[:5], ref_joint_pos[11:]])
    # joint_pos = joints_qpos
    joint_pos = jp.concatenate([joints_qpos[:5], joints_qpos[9:]])

    ref_joint_vels = reference_frame[joint_vels_slice_start:joint_vels_slice_end]
    # remove neck head and antennas
    ref_joint_vels = jp.concatenate([ref_joint_vels[:5], ref_joint_vels[11:]])
    # joint_vel = joints_qvel
    joint_vel = jp.concatenate([joints_qvel[:5], joints_qvel[9:]])

    # ref_left_toe_pos = reference_frame[left_toe_pos_slice_start:left_toe_pos_slice_end]
    # ref_right_toe_pos = reference_frame[right_toe_pos_slice_start:right_toe_pos_slice_end]

    ref_foot_contacts = reference_frame[
        foot_contacts_slice_start:foot_contacts_slice_end
    ]

    # reward
    # torso_pos_rew = jp.exp(-200.0 * jp.sum(jp.square(base_pos[:2] - ref_base_pos[:2]))) * w_torso_pos

    # real quaternion angle doesn't have the expected  effect, switching back for now
    # torso_orientation_rew = jp.exp(-20 * self.quaternion_angle(base_orientation, ref_base_orientation_quat)) * w_torso_orientation

    # TODO ignore yaw here, we just want xy orientation
    torso_orientation_rew = (
        jp.exp(-20.0 * jp.sum(jp.square(base_orientation - ref_base_orientation_quat)))
        * w_torso_orientation
    )

    lin_vel_xy_rew = (
        jp.exp(-8.0 * jp.sum(jp.square(base_lin_vel[:2] - ref_base_lin_vel[:2])))
        * w_lin_vel_xy
    )
    lin_vel_z_rew = (
        jp.exp(-8.0 * jp.sum(jp.square(base_lin_vel[2] - ref_base_lin_vel[2])))
        * w_lin_vel_z
    )

    ang_vel_xy_rew = (
        jp.exp(-2.0 * jp.sum(jp.square(base_ang_vel[:2] - ref_base_ang_vel[:2])))
        * w_ang_vel_xy
    )
    ang_vel_z_rew = (
        jp.exp(-2.0 * jp.sum(jp.square(base_ang_vel[2] - ref_base_ang_vel[2])))
        * w_ang_vel_z
    )

    joint_pos_rew = -jp.sum(jp.square(joint_pos - ref_joint_pos)) * w_joint_pos
    joint_vel_rew = -jp.sum(jp.square(joint_vel - ref_joint_vels)) * w_joint_vel

    ref_foot_contacts = jp.where(
        ref_foot_contacts > 0.5,
        jp.ones_like(ref_foot_contacts),
        jp.zeros_like(ref_foot_contacts),
    )
    contact_rew = jp.sum(contacts == ref_foot_contacts) * w_contact

    reward = (
        lin_vel_xy_rew
        + lin_vel_z_rew
        + ang_vel_xy_rew
        + ang_vel_z_rew
        + joint_pos_rew
        + joint_vel_rew
        + contact_rew
        # + torso_orientation_rew
    )

    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return jp.nan_to_num(reward)
