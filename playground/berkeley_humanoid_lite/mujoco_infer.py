import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.berkeley_humanoid_lite.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = True


class MjInfer(MJInferBase):
    def __init__(
        self, model_path: str, reference_data: str, onnx_model_path: str, standing: bool
    ):
        super().__init__(model_path)

        self.standing = standing

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        if not self.standing:
            self.PRM = PolyReferenceMotion(reference_data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        # Command ranges (vx, vy, yaw_rate)
        self.COMMANDS_RANGE_X = [-0.115, 0.173]
        self.COMMANDS_RANGE_Y = [-0.058, 0.058]
        self.COMMANDS_RANGE_THETA = [-0.385, 0.385]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        # Only 3 command values: [vx, vy, yaw_rate]
        self.commands = np.zeros(3, dtype=np.float32)

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

    def _compute_motor_targets(
        self,
        prev_motor_targets,
        action_command,
    ):
        """Absolute pose target around neutral + per-step slew limiting."""
        proposed = self.default_actuator + action_command * self.action_scale
        delta = proposed - prev_motor_targets
        if USE_MOTOR_SPEED_LIMITS:
            max_step = np.asarray(self.max_motor_velocity, np.float32) * (self.sim_dt * self.decimation)
            delta = np.clip(delta, -max_step, max_step)
        return prev_motor_targets + delta

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)

        imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        R_ws = data.site_xmat[imu_site_id].reshape(3, 3).astype(np.float32)  # world-from-site
        gravity = R_ws.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)      # site frame

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)       # likely a tuple/list
        contacts = np.asarray(contacts, dtype=np.float32)  # make it an array [2]

        # command must be [vx, vy, yaw_rate] -> 3 values
        cmd = np.asarray(command, dtype=np.float32)
        cmd = cmd[:3]  # just in case

        current_ref_motion = self.PRM.get_reference_motion(cmd[0],cmd[1],cmd[2],self.imitation_i)
        ref_actuators = current_ref_motion[0:12]

        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                ref_actuators - self.default_actuator,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                #contacts,
                # ref if not self.standing else np.array([]),
                # [self.imitation_i]
                self.imitation_phase, #Soonho

            ]
        )

        return obs

    def key_callback(self, keycode):
        print(f"key: {keycode}")

        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0.0

        if keycode == 265:  # arrow up
            lin_vel_x = self.COMMANDS_RANGE_X[1]
            # ang_vel = self.COMMANDS_RANGE_THETA[1]
        if keycode == 264:  # arrow down
            lin_vel_x = self.COMMANDS_RANGE_X[0]
            # ang_vel = self.COMMANDS_RANGE_THETA[0]
        if keycode == 263:  # arrow left
            lin_vel_y = self.COMMANDS_RANGE_Y[1]
        if keycode == 262:  # arrow right
            lin_vel_y = self.COMMANDS_RANGE_Y[0]
        if keycode == 81:  # a
            ang_vel = self.COMMANDS_RANGE_THETA[1]
        if keycode == 69:  # e
            ang_vel = self.COMMANDS_RANGE_THETA[0]

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel

    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                counter = 0
                while True:

                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)
                    counter += 1

                    if counter % self.decimation == 0:
                        if not self.standing: 
                            self.imitation_i += 1.0 * self.phase_frequency_factor
                            self.imitation_i = (
                                self.imitation_i % self.PRM.nb_steps_in_period
                            )
                            # print(self.PRM.nb_steps_in_period)
                            # exit()
                            self.imitation_phase = np.array(
                                [
                                    np.cos(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2
                                        * np.pi
                                    ),
                                    np.sin(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2
                                        * np.pi
                                    ),
                                ]
                            )
                        obs = self.get_obs(
                            self.data,
                            self.commands,
                        )
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)

                        # self.action_filter.push(action)
                        # action = self.action_filter.get_filtered_action()

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()
                        self.prev_motor_targets = self.motor_targets.copy()
                        self.motor_targets = self._compute_motor_targets(self.prev_motor_targets, action)
                        self.data.ctrl = self.motor_targets.copy()

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/berkeley_humanoid_lite/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/berkeley_humanoid_lite/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)

    args = parser.parse_args()

    mjinfer = MjInfer(
        args.model_path, args.reference_data, args.onnx_model_path, args.standing
    )
    mjinfer.run()