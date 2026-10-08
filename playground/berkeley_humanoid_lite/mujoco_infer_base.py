import mujoco
import numpy as np
from etils import epath
from playground.berkeley_humanoid_lite import base


class MJInferBase:
    def __init__(self, model_path):

        self.model = mujoco.MjModel.from_xml_string(
            epath.Path(model_path).read_text(), assets=base.get_assets()
        )
        print(model_path)

        self.sim_dt = 0.002
        self.decimation = 10
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        self.num_dofs = self.model.nu

        # Floating base name (assuming a single floating joint)
        self.floating_base_name = [
            self.model.jnt(k).name
            for k in range(0, self.model.njnt)
            if self.model.jnt(k).type == 0
        ][0]

        # Actuator names (these are actuator names, typically matching joint names)
        self.actuator_names = [
            self.model.actuator(k).name for k in range(0, self.model.nu)
        ]

        # All joint names (including floating base and possible backlash joints)
        self.joint_names = [
            self.model.jnt(k).name for k in range(0, self.model.njnt)
        ]

        # ------------------------------------------------------------------
        # FIXED BACKLASH JOINT DETECTION
        #
        # Old behavior:
        #   every joint that is not actuated and not the floating base
        #   was treated as a backlash joint.
        #
        # New behavior:
        #   only joints whose names explicitly indicate "backlash" are
        #   treated as backlash joints. For your robot (no backlash),
        #   this list will be empty.
        #
        # You can customize the heuristic below as needed.
        # ------------------------------------------------------------------
        self.backlash_joint_names = [
            j
            for j in self.joint_names
            if ("backlash" in j.lower()) and (j != self.floating_base_name)
        ]

        # All joint IDs (by name)
        self.all_joint_ids = [self.get_joint_id_from_name(n) for n in self.joint_names]
        self.all_joint_qpos_addr = [
            self.get_joint_addr_from_name(n) for n in self.joint_names
        ]

        # Actuated joint IDs / qpos addr (assuming actuator name == joint name)
        self.actuator_joint_ids = [
            self.get_joint_id_from_name(n) for n in self.actuator_names
        ]
        self.actuator_joint_qpos_addr = [
            self.get_joint_addr_from_name(n) for n in self.actuator_names
        ]

        # Backlash joint IDs / qpos addr (likely empty for your model)
        self.backlash_joint_ids = [
            self.get_joint_id_from_name(n) for n in self.backlash_joint_names
        ]
        self.backlash_joint_qpos_addr = [
            self.get_joint_addr_from_name(n) for n in self.backlash_joint_names
        ]

        # qvel addresses
        self.all_qvel_addr = np.array(
            [self.model.jnt_dofadr[jad] for jad in self.all_joint_ids]
        )
        self.actuator_qvel_addr = np.array(
            [self.model.jnt_dofadr[jad] for jad in self.actuator_joint_ids]
        )

        # Map actuator name -> joint id
        self.actuator_joint_dict = {
            n: self.get_joint_id_from_name(n) for n in self.actuator_names
        }

        # Floating base addresses / id
        self._floating_base_qpos_addr = self.model.jnt_qposadr[
            np.where(self.model.jnt_type == 0)
        ][0]
        self._floating_base_qvel_addr = self.model.jnt_dofadr[
            np.where(self.model.jnt_type == 0)
        ][0]
        self._floating_base_id = self.model.joint(self.floating_base_name).id

        # ------------------------------------------------------------------
        # FIXED: all_joint_no_backlash_ids
        #
        # Meaning: "all joints except backlash joints" (including floating base).
        #
        # The original code mixed backlash joint IDs and qpos indices,
        # which was inconsistent with how this list is used later.
        # ------------------------------------------------------------------
        self.all_joint_no_backlash_ids = [
            jid for jid in self.all_joint_ids if jid not in self.backlash_joint_ids
        ]

        # IMU & sensor setup
        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        self.gyro_addr = self.model.sensor_adr[self.gyro_id]
        self.gyro_dimensions = 3

        self.accelerometer_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer"
        )
        self.accelerometer_dimensions = 3
        self.accelerometer_addr = self.model.sensor_adr[self.accelerometer_id]

        self.linvel_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "local_linvel"
        )
        self.linvel_dimensions = 3

        self.imu_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )

        self.gravity_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "upvector"
        )
        self.gravity_dimensions = 3

        # Initial state from "home" keyframe
        self.init_pos = np.array(
            self.get_all_joints_qpos(self.model.keyframe("home").qpos)
        )  # pose of all the joints (no floating base)
        home_qpos = self.model.keyframe("home").qpos
        self.default_actuator = self.get_actuator_joints_qpos(home_qpos).copy()

        self.motor_targets = self.default_actuator
        self.prev_motor_targets = self.default_actuator

        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.ctrl[:] = self.default_actuator

        self.left_foot_geom_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_bottom"
        )
        self.right_foot_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_bottom"
        )
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )


    # -------------------------
    # ID / address helpers
    # -------------------------

    def get_actuator_id_from_name(self, name: str) -> int:
        """Return the id of a specified actuator"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def get_joint_id_from_name(self, name: str) -> int:
        """Return the id of a specified joint"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def get_joint_addr_from_name(self, name: str) -> int:
        """Return the address of a specified joint"""
        return self.model.joint(name).qposadr

    def get_dof_id_from_name(self, name: str) -> int:
        """Return the id of a specified dof"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_DOF, name)

    # -------------------------
    # Actuated joints helpers
    # -------------------------

    def get_actuator_joint_qpos_from_name(
        self, data: np.ndarray, name: str
    ) -> np.ndarray:
        """Return the qpos of a given actuated joint"""
        addr = self.model.jnt_qposadr[self.actuator_joint_dict[name]]
        return data[addr]

    def get_actuator_joints_addr(self) -> np.ndarray:
        """Return all qpos indices of actuated joints"""
        addr = np.array(
            [self.model.jnt_qposadr[idx] for idx in self.actuator_joint_ids]
        )
        return addr

    def get_actuator_joints_qpos(self, data: np.ndarray) -> np.ndarray:
        """Return all qpos of actuated joints"""
        return data[self.get_actuator_joints_addr()]

    def set_actuator_joints_qpos(
        self, new_qpos: np.ndarray, qpos: np.ndarray
    ) -> np.ndarray:
        """Set the qpos only for the actuated joints (omit backlash joints)"""
        qpos[self.get_actuator_joints_addr()] = new_qpos
        return qpos

    def get_actuator_joints_qvel(self, data: np.ndarray) -> np.ndarray:
        """Return all qvel of actuated joints"""
        return data[self.actuator_qvel_addr]

    def set_actuator_joints_qvel(
        self, new_qvel: np.ndarray, qvel: np.ndarray
    ) -> np.ndarray:
        """Set the qvel only for the actuated joints (omit backlash joints)"""
        qvel[self.actuator_qvel_addr] = new_qvel
        return qvel

    # -------------------------
    # Floating base helpers
    # -------------------------

    def get_floating_base_qpos(self, data: np.ndarray) -> np.ndarray:
        return data[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 7]

    def get_floating_base_qvel(self, data: np.ndarray) -> np.ndarray:
        return data[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6]

    def set_floating_base_qpos(
        self, new_qpos: np.ndarray, qpos: np.ndarray
    ) -> np.ndarray:
        qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 7] = (
            new_qpos
        )
        return qpos

    def set_floating_base_qvel(
        self, new_qvel: np.ndarray, qvel: np.ndarray
    ) -> np.ndarray:
        qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6] = (
            new_qvel
        )
        return qvel

    # -------------------------
    # Backlash exclusion / all joints
    # -------------------------

    def exclude_backlash_joints_addr(self) -> np.ndarray:
        """Return all qpos indices of joints excluding backlash joints"""
        addr = np.array(
            [self.model.jnt_qposadr[idx] for idx in self.all_joint_no_backlash_ids]
        )
        return addr

    def get_all_joints_addr(self) -> np.ndarray:
        """Return all qpos indices of all joints"""
        addr = np.array([self.model.jnt_qposadr[idx] for idx in self.all_joint_ids])
        return addr

    def get_all_joints_qpos(self, data: np.ndarray) -> np.ndarray:
        """Return all qpos of all joints"""
        return data[self.get_all_joints_addr()]

    def get_all_joints_qvel(self, data: np.ndarray) -> np.ndarray:
        """Return all qvel of all joints"""
        return data[self.all_qvel_addr]

    def get_joints_nobacklash_qpos(self, data: np.ndarray) -> np.ndarray:
        """Return qpos of all non-backlash joints (including floating base)"""
        return data[self.exclude_backlash_joints_addr()]

    def set_complete_qpos_from_joints(
        self, no_backlash_qpos: np.ndarray, full_qpos: np.ndarray
    ) -> np.ndarray:
        """
        In the case of backlash joints, we want to ignore them (remove them)
        but we still need to set the complete state including them.
        """
        full_qpos[self.exclude_backlash_joints_addr()] = no_backlash_qpos
        return np.array(full_qpos)

    # -------------------------
    # Sensors
    # -------------------------

    def get_sensor(self, data, name, dimensions):
        i = self.model.sensor_name2id(name)
        return data.sensordata[i : i + dimensions]

    def get_gyro(self, data):
        return data.sensordata[self.gyro_addr : self.gyro_addr + self.gyro_dimensions]

    def get_accelerometer(self, data):
        return data.sensordata[
            self.accelerometer_addr : self.accelerometer_addr
            + self.accelerometer_dimensions
        ]

    def get_linvel(self, data):
        return data.sensordata[self.linvel_id : self.linvel_id + self.linvel_dimensions]

    def get_gravity(self, data):
        return data.sensordata[
            self.gravity_id : self.gravity_id + self.gravity_dimensions
        ]

    # -------------------------
    # Contacts
    # -------------------------
    def check_contact(self, data, geom1_id: int, geom2_id: int) -> bool:
        """Return True if geom1 and geom2 are in contact."""
        for i in range(data.ncon):
            c = data.contact[i]
            if (
                (c.geom1 == geom1_id and c.geom2 == geom2_id)
                or (c.geom1 == geom2_id and c.geom2 == geom1_id)
            ):
                return True
        return False
    
    
    def get_feet_contacts(self, data):
        left_contact = self.check_contact(
            data, self.left_foot_geom_id, self.floor_geom_id
        )
        right_contact = self.check_contact(
            data, self.right_foot_geom_id, self.floor_geom_id
        )
        return left_contact, right_contact
    
