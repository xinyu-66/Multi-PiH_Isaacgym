import hydra
import numpy as np
import os
import torch

from isaacgym import gymapi
from isaacgymenvs.tasks.peg_in_hole.peg_in_hole_3circle_base import PegInHoleBase
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp


class PegInHoleEnv(PegInHoleBase):
    def __init__(
            self, 
            cfg, 
            rl_device, 
            sim_device, 
            graphics_device_id, 
            headless, 
            virtual_screen_capture, 
            force_render
        ):
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""

        self._get_env_yaml_params()

        super().__init__(
            cfg, 
            rl_device, 
            sim_device, 
            graphics_device_id, 
            headless, 
            virtual_screen_capture, 
            force_render
        )

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()
    
    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        # cs = hydra.core.config_store.ConfigStore.instance()
        # cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)

        # config_path = 'task/FactoryEnvpeghole.yaml'  # relative to Hydra search path (cfg dir)
        # self.cfg_env = hydra.compose(config_name=config_path)
        # self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

        asset_info_path = '../../assets/factory/yaml/asset_info_peg_in_hole.yaml'
        self.asset_info_peg_hole = hydra.compose(config_name=asset_info_path)
        self.asset_info_peg_hole = self.asset_info_peg_hole['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        self.print_sdf_warning()
        franka_asset, table_asset, table_stand_asset = self.import_franka_assets()
        peg_asset, hole_asset = self._import_env_assets()
        self._create_actors(lower, upper, num_per_row, franka_asset, peg_asset, hole_asset, table_asset)

    def _import_env_assets(self):
        """Set peg and hole asset options. Import assets."""

        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'factory', 'urdf')

        peg_options = gymapi.AssetOptions()
        peg_options.flip_visual_attachments = False
        peg_options.fix_base_link = False
        peg_options.thickness = 0.0  # default = 0.02
        peg_options.armature = 0.0  # default = 0.0
        peg_options.use_physx_armature = True
        peg_options.linear_damping = 0.0  # default = 0.0
        peg_options.max_linear_velocity = 1000.0  # default = 1000.0
        peg_options.angular_damping = 0.0  # default = 0.5
        peg_options.max_angular_velocity = 64.0  # default = 64.0
        peg_options.disable_gravity = False
        peg_options.enable_gyroscopic_forces = True
        peg_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        peg_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            peg_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        hole_options = gymapi.AssetOptions()
        hole_options.flip_visual_attachments = False
        hole_options.fix_base_link = True
        hole_options.thickness = 0.0  # default = 0.02
        hole_options.armature = 0.0  # default = 0.0
        hole_options.use_physx_armature = True
        hole_options.linear_damping = 0.0  # default = 0.0
        hole_options.max_linear_velocity = 1000.0  # default = 1000.0
        hole_options.angular_damping = 0.0  # default = 0.5
        hole_options.max_angular_velocity = 64.0  # default = 64.0
        hole_options.disable_gravity = False
        hole_options.enable_gyroscopic_forces = True
        hole_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        hole_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            hole_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        peg_assets = []
        hole_assets = []
        peg_file = "three_circle_peg.urdf"
        hole_file = "three_circle_hole.urdf"
        # for subassembly in self.cfg_env.env.desired_subassemblies:
        #     components = list(self.asset_info_peg_hole[subassembly])
        #     peg_file = self.asset_info_peg_hole[subassembly][components[0]]['urdf_path'] + '.urdf'
        #     hole_file = self.asset_info_peg_hole[subassembly][components[1]]['urdf_path'] + '.urdf'
        #     peg_options.density = self.cfg_env.env.peg_hole_density
        #     hole_options.density = self.cfg_env.env.peg_hole_density
        #     peg_asset = self.gym.load_asset(self.sim, urdf_root, peg_file, peg_options)
        #     hole_asset = self.gym.load_asset(self.sim, urdf_root, hole_file, hole_options)
        #     peg_assets.append(peg_asset)
        #     hole_assets.append(hole_asset)

        peg_asset = self.gym.load_asset(self.sim, urdf_root, peg_file, peg_options)
        hole_asset = self.gym.load_asset(self.sim, urdf_root, hole_file, hole_options)

        peg_assets.append(peg_asset)
        hole_assets.append(hole_asset)

        return peg_assets, hole_assets
    
    def _create_actors(self, lower, upper, num_per_row, franka_asset, peg_assets, hole_assets, table_asset, table_stand_asset):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        table_thickness = 0.05
        table_stand_height = 0.1

        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_pose = gymapi.Transform()
        table_pose.p = [0.0, 0.0, 1.0]
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_stand_pose = gymapi.Transform()
        table_stand_pose.p = gymapi.Vec3(-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2)
        table_stand_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        self.env_ptrs = []
        self.franka_handles = []
        self.peg_handles = []
        self.hole_handles = []
        self.table_handles = []
        self.table_stand_handles = []
        self.shape_ids = []
        self.franka_actor_ids_sim = []  # within-sim indices
        self.peg_actor_ids_sim = []  # within-sim indices
        self.hole_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        self.table_stand_actor_ids_sim = []
        actor_count = 0

        self.peg_heights = []
        self.hole_heights = []
        self.peg_widths_max = []
        self.hole_widths = []
        self.hole_head_heights = []
        self.hole_shank_lengths = []
        self.thread_pitches = []

        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []

        for i in range(self.num_envs):

            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, 'franka', i + self.num_envs,
                                                      0, 0)
            else:
                franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, 'franka', i, 0, 0)
            self.franka_actor_ids_sim.append(actor_count)
            actor_count += 1

            j = 0
            # subassembly = self.cfg_env.env.desired_subassemblies[j]
            # components = list(self.asset_info_peg_hole[subassembly])

            peg_start_pose = gymapi.Transform()
            peg_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
            peg_start_pose.r = gymapi.Quat(1.0, 0.0, 0.0, 0.0)

            peg_handle = self.gym.create_actor(env_ptr, peg_assets[j], peg_start_pose, 'peg', i, 0, 0)
            self.peg_actor_ids_sim.append(actor_count)
            actor_count += 1
            peg_height = 0.09
            self.peg_heights.append(peg_height)

            hole_start_pose = gymapi.Transform()
            hole_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
            hole_start_pose.r = gymapi.Quat(1.0, 0.0, 0.0, 0.0)

            hole_handle = self.gym.create_actor(env_ptr, hole_assets[j], hole_start_pose, 'hole', i, 0, 0)
            self.hole_actor_ids_sim.append(actor_count)
            actor_count += 1
            hole_height = 0.08
            self.hole_heights.append(hole_height)

            # thread_pitch = self.asset_info_peg_hole[subassembly]['thread_pitch']

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, 'table', i, 0, 0)
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

            table_stand_handle = self.gym.create_actor(env_ptr, table_stand_asset, table_pose, 'table_stand', i, 0, 0)
            self.table_stand_actor_ids_sim.append(actor_count)
            actor_count += 1

            link7_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_link7', gymapi.DOMAIN_ACTOR)
            hand_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand', gymapi.DOMAIN_ACTOR)
            left_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                  gymapi.DOMAIN_ACTOR)
            right_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_rightfinger',
                                                                   gymapi.DOMAIN_ACTOR)
            grip_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_grip_site',
                                                                   gymapi.DOMAIN_ACTOR)
            self.shape_ids = [link7_id, hand_id, left_finger_id, right_finger_id, grip_id]

            franka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_handle)
            for shape_id in self.shape_ids:
                franka_shape_props[shape_id].friction = self.cfg_base.env.franka_friction
                franka_shape_props[shape_id].rolling_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].torsion_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].restitution = 0.0  # default = 0.0
                franka_shape_props[shape_id].compliance = 0.0  # default = 0.0
                franka_shape_props[shape_id].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, franka_handle, franka_shape_props)

            peg_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, peg_handle)
            peg_shape_props[0].friction = self.cfg_env.env.peg_hole_friction
            peg_shape_props[0].rolling_friction = 0.0  # default = 0.0
            peg_shape_props[0].torsion_friction = 0.0  # default = 0.0
            peg_shape_props[0].restitution = 0.0  # default = 0.0
            peg_shape_props[0].compliance = 0.0  # default = 0.0
            peg_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, peg_handle, peg_shape_props)

            hole_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hole_handle)
            hole_shape_props[0].friction = self.cfg_env.env.peg_hole_friction
            hole_shape_props[0].rolling_friction = 0.0  # default = 0.0
            hole_shape_props[0].torsion_friction = 0.0  # default = 0.0
            hole_shape_props[0].restitution = 0.0  # default = 0.0
            hole_shape_props[0].compliance = 0.0  # default = 0.0
            hole_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, hole_handle, hole_shape_props)

            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0  # default = 0.0
            table_shape_props[0].torsion_friction = 0.0  # default = 0.0
            table_shape_props[0].restitution = 0.0  # default = 0.0
            table_shape_props[0].compliance = 0.0  # default = 0.0
            table_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
            
            for i in range(self.num_franka_dofs):
                franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
                if self.physics_engine == gymapi.SIM_PHYSX:
                    franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                    franka_dof_props['damping'][i] = franka_dof_damping[i]
                # else:
                #     franka_dof_props['stiffness'][i] = 7000.0
                #     franka_dof_props['damping'][i] = 50.0

                self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
                self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
                self._franka_effort_limits.append(franka_dof_props['effort'][i])

            self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
            self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
            self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
            self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
            self.franka_dof_speed_scales[[7, 8]] = 0.1
            franka_dof_props['effort'][7] = 200
            franka_dof_props['effort'][8] = 200

            # self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)
            self.franka_handles.append(franka_handle)
            self.peg_handles.append(peg_handle)
            self.hole_handles.append(hole_handle)
            self.table_handles.append(table_handle)
            self.table_stand_handles.append(table_stand_handle)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(self.franka_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.peg_actor_ids_sim = torch.tensor(self.peg_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.hole_actor_ids_sim = torch.tensor(self.hole_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.peg_actor_id_env = self.gym.find_actor_index(env_ptr, 'peg', gymapi.DOMAIN_ENV)
        self.hole_actor_id_env = self.gym.find_actor_index(env_ptr, 'hole', gymapi.DOMAIN_ENV)

        # For extracting body pos/quat, force, and Jacobian
        self.peg_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, peg_handle, 'peg', gymapi.DOMAIN_ENV)
        self.hole_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, hole_handle, 'hole', gymapi.DOMAIN_ENV)
        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand',
                                                                     gymapi.DOMAIN_ENV)
        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                            gymapi.DOMAIN_ENV)
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                             'panda_rightfinger', gymapi.DOMAIN_ENV)
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                   'panda_fingertip_centered',
                                                                                   gymapi.DOMAIN_ENV)

        # For computing body COM pos
        self.peg_heights = torch.tensor(self.peg_heights, device=self.device).unsqueeze(-1)
        self.hole_head_heights = torch.tensor(self.hole_head_heights, device=self.device).unsqueeze(-1)

        # For setting initial state
        self.peg_widths_max = torch.tensor(self.peg_widths_max, device=self.device).unsqueeze(-1)
        self.hole_shank_lengths = torch.tensor(self.hole_shank_lengths, device=self.device).unsqueeze(-1)

        # For defining success or failure
        self.hole_widths = torch.tensor(self.hole_widths, device=self.device).unsqueeze(-1)
        self.thread_pitches = torch.tensor(self.thread_pitches, device=self.device).unsqueeze(-1)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        self._peg_grippoint_state = self._rigid_body_state[:, self.handles["peg_point_handle"], :]
        self._hole_setpoint_state = self._rigid_body_state[:, self.handles["hole_point_handle"], :]
        self._peg_state = self._rigid_body_state[:, self.handles["peg_point_handle"], :]
        self._hole_state = self._rigid_body_state[:, self.handles["three_circle_hole"], :]


        self.nut_pos = self.root_pos[:, self.nut_actor_id_env, 0:3]
        self.nut_quat = self.root_quat[:, self.nut_actor_id_env, 0:4]
        self.nut_linvel = self.root_linvel[:, self.nut_actor_id_env, 0:3]
        self.nut_angvel = self.root_angvel[:, self.nut_actor_id_env, 0:3]

        self.bolt_pos = self.root_pos[:, self.bolt_actor_id_env, 0:3]
        self.bolt_quat = self.root_quat[:, self.bolt_actor_id_env, 0:4]

        self.nut_force = self.contact_force[:, self.nut_body_id_env, 0:3]

        self.bolt_force = self.contact_force[:, self.bolt_body_id_env, 0:3]

        self.nut_com_pos = fc.translate_along_local_z(pos=self.nut_pos,
                                                      quat=self.nut_quat,
                                                      offset=self.bolt_head_heights + self.nut_heights * 0.5,
                                                      device=self.device)
        self.nut_com_quat = self.nut_quat  # always equal
        self.nut_com_linvel = self.nut_linvel + torch.cross(self.nut_angvel,
                                                            (self.nut_com_pos - self.nut_pos),
                                                            dim=1)
        self.nut_com_angvel = self.nut_angvel  # always equal

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        self.nut_com_pos = fc.translate_along_local_z(pos=self.nut_pos,
                                                      quat=self.nut_quat,
                                                      offset=self.bolt_head_heights + self.nut_heights * 0.5,
                                                      device=self.device)
        self.nut_com_linvel = self.nut_linvel + torch.cross(self.nut_angvel,
                                                            (self.nut_com_pos - self.nut_pos),
                                                            dim=1)