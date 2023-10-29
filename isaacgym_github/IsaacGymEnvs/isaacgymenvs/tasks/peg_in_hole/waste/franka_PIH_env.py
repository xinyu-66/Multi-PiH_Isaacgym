# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask
from .franka_PIH_base import FrankaPIHBase

class FrankaPIHEnv(FrankaPIHBase):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        super().__init__(
            cfg, 
            rl_device, 
            sim_device, 
            graphics_device_id, 
            headless, 
            virtual_screen_capture, 
            force_render)

        # Reset all environments
        # self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # # Refresh tensors
        # self._refresh()

    def create_envs(self, num_envs, spacing, num_per_row):

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        franka_asset, table_asset, table_stand_asset, table_thickness, table_stand_height = self.import_franka_assets()
        peg_asset, hole_asset = self._import_env_assets()
        self._create_actors(lower, upper, num_per_row, franka_asset, table_asset, table_stand_asset, peg_asset, hole_asset, table_thickness, table_stand_height)

    def import_franka_assets(self):
        """Set Franka and table asset options. Import assets."""

        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'urdf', 'franka_description', 'robots')
        franka_file = 'franka_panda_gripper.urdf'

        self.franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        self.franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # load franka asset
        franka_options = gymapi.AssetOptions()
        franka_options.flip_visual_attachments = True
        franka_options.fix_base_link = True
        franka_options.collapse_fixed_joints = False
        franka_options.disable_gravity = True
        franka_options.thickness = 0.001  # default = 0.02
        franka_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        franka_options.density = 1000.0  # default = 1000.0
        franka_options.armature = 0.01  # default = 0.0
        franka_options.use_physx_armature = True
        franka_options.enable_gyroscopic_forces = True
        franka_options.use_mesh_materials = True

        franka_asset = self.gym.load_asset(self.sim, urdf_root, franka_file, franka_options)
        # if self.cfg_base.sim.add_damping:
        #     franka_options.linear_damping = 1.0  # default = 0.0; increased to improve stability
        #     franka_options.max_linear_velocity = 1.0  # default = 1000.0; reduced to prevent CUDA errors
        #     franka_options.angular_damping = 5.0  # default = 0.5; increased to improve stability
        #     franka_options.max_angular_velocity = 2 * math.pi  # default = 64.0; reduced to prevent CUDA errors
        # else:
        #     franka_options.linear_damping = 0.0  # default = 0.0
        #     franka_options.max_linear_velocity = 1000.0  # default = 1000.0
        #     franka_options.angular_damping = 0.5  # default = 0.5
        #     franka_options.max_angular_velocity = 64.0  # default = 64.0
        # if self.cfg_base.mode.export_scene:
        #     franka_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

         # Create table asset
        self.table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        self.table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        return franka_asset, table_asset, table_stand_asset, table_thickness, table_stand_height

    def _import_env_assets(self):
        """Set peg and hole asset options. Import assets."""

        self.peg_size = 0.080
        self.hole_size = 0.090

        self.peg_color = gymapi.Vec3(0.6, 0.1, 0.0)
        self.hole_color = gymapi.Vec3(0.0, 0.4, 0.1)

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
        # if self.cfg_base.mode.export_scene:
        #     peg_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

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
        # if self.cfg_base.mode.export_scene:
        #     hole_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

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

    def _create_actors(self, lower, upper, num_per_row, franka_asset, table_asset, table_stand_asset, peg_assets, hole_assets, table_thickness, table_stand_height):
        
        self.franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        self.franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)


        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)


        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = self.franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = self.franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

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

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*self.table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(self.table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*self.table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        peg_start_pose = gymapi.Transform()
        peg_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        peg_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        hole_start_pose = gymapi.Transform()
        hole_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        hole_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 100     # 1 for table, table stand, peg, hole
        max_agg_shapes = num_franka_shapes + 100     # 1 for table, table stand, peg, hole

        self.frankas = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._peg_id = self.gym.create_actor(env_ptr, peg_assets[0], peg_start_pose, "peg", i, 2, 0)
            self._hole_id = self.gym.create_actor(env_ptr, hole_assets[0], hole_start_pose, "hole", i, 4, 0)
            
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._peg_id, 0, gymapi.MESH_VISUAL, self.peg_color)
            self.gym.set_rigid_body_color(env_ptr, self._hole_id, 0, gymapi.MESH_VISUAL, self.hole_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_peg_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_hole_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()