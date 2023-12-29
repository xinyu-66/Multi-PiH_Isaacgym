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
import matplotlib.pyplot as plt

from isaacgym import gymtorch
from isaacgym import gymapi
import pandas as pd

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask
from .franka_PIH_base import FrankaPIHBase

class FrankaPIHTaskInsert(FrankaPIHBase):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        super().__init__(
            cfg, 
            rl_device, 
            sim_device, 
            graphics_device_id, 
            headless, 
            virtual_screen_capture, 
            force_render)

        # self.start_row = 0 #only for wring data into excel file
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self.gripper_bool = torch.zeros_like(self.reset_buf)
        self._refresh()

    def create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        # Create peg and hole asset, peg firstly
        self.peg_size = 0.050
        self.hole_size = 0.070

        peg_color = gymapi.Vec3(0.6, 0.1, 0.0)
        hole_color = gymapi.Vec3(0.0, 0.4, 0.1)

        peg_opts = gymapi.AssetOptions()
        hole_opts = gymapi.AssetOptions()

        peg_asset_file = "factory/urdf/three_circle_peg.urdf"
        peg_asset_file = franka_asset_file = self.cfg["env"]["asset"].get("assetFileNamePeg", peg_asset_file)
        
        hole_asset_file = "factory/urdf/three_circle_hole.urdf"
        hole_asset_file = franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameHole", hole_asset_file)
        
        peg_asset = self.gym.load_asset(self.sim, asset_root, peg_asset_file, peg_opts)
        hole_asset = self.gym.load_asset(self.sim, asset_root, hole_asset_file, hole_opts)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
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
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]
        self.reward_settings["grippoint_height"] = 0.055

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
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
        max_agg_bodies = num_franka_bodies + 10     # 1 for table, table stand, peg, hole
        max_agg_shapes = num_franka_shapes + 10     # 1 for table, table stand, peg, hole

        self.frankas = []
        self.envs = []

        self.camera_handles = []
        cam_tensors = []

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
            self._peg_id = self.gym.create_actor(env_ptr, peg_asset, peg_start_pose, "peg", i, 2, 0)
            self._hole_id = self.gym.create_actor(env_ptr, hole_asset, hole_start_pose, "hole", i, 4, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._peg_id, 0, gymapi.MESH_VISUAL, peg_color)
            self.gym.set_rigid_body_color(env_ptr, self._hole_id, 0, gymapi.MESH_VISUAL, hole_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            ########################## Camera ##########################
            if self.add_camera or self.only_camera:
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                camera_properties = gymapi.CameraProperties()
                camera_properties.width = self.img_width
                camera_properties.height = self.img_height
                camera_properties.enable_tensors = True

                # Set a fixed position and look-target for the first camera
                # position and target location are in the coordinate frame of the environment
                handle = self.gym.create_camera_sensor(env_ptr, camera_properties)
                camera_position = gymapi.Vec3(0.5, 0.6, 1.4) #(0.4, 0, 1.6)(0.4, 0, 1.6) best updown:(0.4, 0, 1.7) betterside: (0.5, 0.6, 1.4)
                camera_target = gymapi.Vec3(0, 0, 1.2) #(0, 0, 1.2)(0, 0, 0.2) best updown: (0, 0, 0.0) betterside: (0, 0, 1.2)
                self.gym.set_camera_location(handle, env_ptr, camera_position, camera_target)
                self.camera_handles.append(handle)

                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, handle, gymapi.IMAGE_COLOR)
                torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
                torch_cam_tensor = torch_cam_tensor[..., :3]
                # print("Got camera tensor with shape", torch_cam_tensor.shape)
                # print("Got camera tensor with shape", torch_cam_tensor)
                cam_tensors.append(torch_cam_tensor)
                print(f"Created the {i} env with the camera {handle}.")
            else:
                # print(self.only_camera)
                print(f"Created the {i} env.")

        print("Collect ", len(cam_tensors), "images from ", len(self.camera_handles) ," camera tensors")
        # Setup init state buffer
        self._init_peg_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_hole_state = torch.zeros(self.num_envs, 13, device=self.device)
        if self.add_camera:
            cam_tensor = torch.stack(cam_tensors, dim=0).float()
            batch = cam_tensor.shape[0]
            cam_tensor = cam_tensor.permute(0, 3, 1, 2)
            print("cam_tensor", cam_tensor.shape)
            cam_tensor = cam_tensor.reshape(batch, -1)
            self.images = cam_tensor
            print("INITself.images.shape", self.images.shape, "without progress created")
        # Setup data
        self.init_data()

    def compute_reward(self, actions):
        self.gripper_bool = torch.zeros_like(self.reset_buf)
        self.rew_buf[:], self.reset_buf[:], self.gripper_bool[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()
        # obs = 22
        obs = ["peg_quat", "peg_pos", "peg_to_hole_pos", "eef_pos", "eef_quat", "hand_force"]# "peg_to_hole_quat", 
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        
        # obs = 15
        # obs = ["q", "hole_pos", "hand_force"]
        # obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        
        # obs = 12
        # obs = ["q", "hole_pos"]

        #obs = 10
        # obs = ["peg_pos", "peg_quat", "hole_pos"]# "peg_to_hole_quat",

        # obs = 13 in model 15
        # obs = ["eef_pos", "eef_quat", "hole_pos", "hand_force"]# "peg_to_hole_quat", 
        # obs += ["q_gripper"] if self.control_type == "osc" else ["q"]

        # obs = 10
        # obs = ["eef_pos", "eef_quat", "hole_pos"]# "peg_to_hole_quat", 

        if self.add_camera and not self.store_dataset:
            # obs = ["input_images", "hand_force", "q"]
            obs = ["input_images", "q", "hole_pos", "hand_force"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        # print(self.obs_buf.shape)
        # print(self.obs_buf[:, :-12])
        # print(self.states["q"])
        # print(self.states["hand_force"])

        # print("compute_observations", self.obs_buf)
        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}
        
        return self.obs_buf
    
    def _reset_franka(self, env_ids):

        # Reset agent
        # reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        # pos = tensor_clamp(
        #     self.franka_default_dof_pos.unsqueeze(0) +
        #     self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
        #     self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)
        
        pos = tensor_clamp(
        self.franka_default_dof_pos.unsqueeze(0),
        self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
        
    def _reset_obj(self, env_ids):
        # Reset cubes, sampling cube B first, then A
        self._reset_init_cube_state(cube='B', env_ids=env_ids, check_valid=False)
        self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=False)

        # Write these new init states to the sim states
        self._peg_state[env_ids] = self._init_peg_state[env_ids]
        self._hole_state[env_ids] = self._init_hole_state[env_ids]

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))


    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self._reset_franka(env_ids)
        self._reset_obj(env_ids)

        self.disable_gravity()
        for _ in range(10):
            # print(self._eef_state[:, :3])
            # print(self._dof_state[..., 0])

            self._init_peg_state[env_ids, :3] = self._eef_state[env_ids, :3]
            self._init_peg_state[env_ids, 2] -= 0.055
            # self._init_peg_state[env_ids, 0] -= 0.01
            # Initialize the hole's position to track the peg's position
            self._init_hole_state[env_ids, :2] = self._eef_state[env_ids, :2] 
            # + 0.01 * (torch.rand_like(self._eef_state[env_ids, :2]))

            self._peg_state[env_ids] = self._init_peg_state[env_ids]
            if self.add_camera and self.only_insert:
                self._hole_state[env_ids] = self._init_hole_state[env_ids]
            
            multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))
            
            self.gym.simulate(self.sim)
            self.render()

        self.enable_gravity()
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if cube.lower() == 'a':
            this_cube_state_all = self._init_peg_state
            other_cube_state = self._init_hole_state[env_ids, :]
            cube_heights = self.states["peg_size"]
        elif cube.lower() == 'b':
            this_cube_state_all = self._init_hole_state
            other_cube_state = self._init_peg_state[env_ids, :]
            cube_heights = self.states["peg_size"]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        min_dists = (self.states["peg_size"] + self.states["hole_size"])[env_ids] * np.sqrt(2)

        # We scale the min dist by 2 so that the cubes aren't too close together
        min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_cube_x_state = torch.tensor(self._table_surface_pos[0], device=self.device, dtype=torch.float32)
        centered_cube_y_state = torch.tensor(self._table_surface_pos[1], device=self.device, dtype=torch.float32)
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._table_surface_pos[2]

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 3] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid:
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # Sample x y values
                sampled_cube_state[active_idx, :2] = centered_cube_xy_state + \
                                                     2.0 * self.start_position_noise * (
                                                             torch.rand_like(sampled_cube_state[active_idx, :2]) - 0.5)
                # Check if sampled values are valid
                cube_dist = torch.linalg.norm(sampled_cube_state[:, :2] - other_cube_state[:, :2], dim=-1)
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling cube locations was unsuccessful! ):"
        else:
            # We just directly sample
            sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)
            # sampled_cube_state[:, 0] = centered_cube_x_state.unsqueeze(0) + \
            #                                   2.0 *  * (torch.rand(num_resets, device=self.device) - 0.5)
        
            # sampled_cube_state[:, 1] = centered_cube_y_state.unsqueeze(0) + \
            #                                   2.0 *  * (torch.rand(num_resets, device=self.device) - 0.5)


        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        this_cube_state_all[env_ids, :] = sampled_cube_state

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # print("u_arm.shape before ", u_arm[0])

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        
        # print("u_arm.shape after ", u_arm[0])
        self._arm_control[:, :] = u_arm

        # print("dof_state[0]before ", self._q[0, :7])

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                      self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                      self.franka_dof_lower_limits[-1].item())
        
        # Write gripper command to appropriate tensor buffer, open the gripper when after inserting the peg
        self.gripper_bool_fingers = torch.stack((self.gripper_bool, self.gripper_bool), 1)
        self._gripper_control[:, :] = torch.where(self.gripper_bool_fingers > 0, torch.ones_like(self._gripper_control) * 0.08, torch.ones_like(self._gripper_control) * 0.01)

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
        
        # print("PREPREPRPERPEPRprint(self.actions)", self.actions)
        # print("PREPREPRPERPEPRnp.array(self._peg_grippoint_state[0, :7].cpu())", np.array(self._peg_grippoint_state[0, :7].cpu()), "with progress", self.progress_buf[:20])

        # the first observation is in the _refresh(), and the action will generated after env.resert and before pre_step(), so 
        # collect data and actions here is the best
        if self.store_dataset:
            if self.progress_buf[63] == 0:
                self.flag_epoch += 1
                print("****************************************************************************")
                print("self.flag_epoch += 1, now is: ", self.flag_epoch)
                print("****************************************************************************")

            rgb_names = []
            if self.progress_buf[0] % 1 == 0:
                peg_pos_quat = np.array(self._peg_grippoint_state[:, :7].cpu()) # position(3) + orientation(4)
                hole_pos_quat = np.array(self._hole_state[:, :7].cpu()) # position(3) + orientation(4)

                dof_state = np.array(self._q[:, :7].cpu()) # DOF(7)
                dof_speed = np.array(self._qd[:, :7].cpu()) # DOF(7)

                left_finger_force = np.array(self.left_finger_force.cpu()) # xyz(3)
                right_finger_force = np.array(self.right_finger_force.cpu()) # xyz(3)
                hand_force = np.array(self.hand_force.cpu()) # xyz(3)

                insert_flag = np.array(self.gripper_bool.cpu()) # bool(1)
                insert_flag = insert_flag.reshape(-1, 1)

                action = np.array(self.actions.cpu()) # delta eef(6) + gripper_bool(1)
                action = action[:, :-1]

                # print("action[0]", action[0])
                # print("dof_state[0]", dof_state[0])
                #all array has first dimension of num_envs
                data = np.hstack((peg_pos_quat, hole_pos_quat, dof_state, dof_speed, left_finger_force, right_finger_force, hand_force, action, insert_flag)) # dimentation:44
                print("##########################################################")
                print("Collected data in the environment: ", data.shape)

                path = "data.xlsx"
                #self.num_envs
                for i in range(63,64):
                    # The gym utility to write images to disk is recommended only for RGB images.
                    root = "graphics_images"
                    rgb_name = "rgb_env%d_progress%d in epoch %d.png" % (i, self.progress_buf[i], self.flag_epoch)
                    file_path = os.path.join(root, rgb_name)
                    rgb_names.append(rgb_name)
                    self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR, file_path)
                    print("RGB images stored ")

                if os.path.exists(path):
                    pre_data = pd.read_excel(path, engine='openpyxl', index_col=0)
                    print("Writter open and already stored data in excel: ", pre_data.shape)
                    writer = pd.ExcelWriter(path, engine='openpyxl')
                    pre_data.to_excel(writer, "sheet_1", float_format='%.2f', startrow=0)#columns=["peg_x", "peg_y", "peg_z"]
                    df = pd.DataFrame(data[i, :].reshape(1, -1), index=rgb_names)
                    df.to_excel(writer, "sheet_1", float_format='%.2f', startrow=pre_data.shape[0]+1, header=None)#columns=["peg_x", "peg_y", "peg_z"]
                    writer.close()
                    # assert (pre_data.shape[0] + 1) / 166 != 100, "Sampling cube locations was unsuccessful! ):"
                    print("Written finished and closed")

                else:
                    # np.savetxt("my_data.csv", data, delimiter=",", fmt="%.2f")
                    # print("self.start_row", self.start_row)
                    df = pd.DataFrame(data[i, :].reshape(1, -1), index=rgb_names)
                    writer = pd.ExcelWriter(path, engine='openpyxl')
                    df.to_excel(writer, "sheet_1", float_format='%.2f', startrow=0)#columns=["peg_x", "peg_y", "peg_z"]
                    writer.close()
                    print("Written finished and closed")

    def post_physics_step(self):

        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # print("self.left_finger_force", self.left_finger_force[index, :])
        # print("self.right_finger_force", self.right_finger_force[index, :])
        # print("self.hand_force", self.hand_force[index, :])

        self.compute_observations()
        self.compute_reward(self.actions)
        # print("dof_state[0]after ", self._q[0, :7])

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            peg_pos = self.states["peg_pos"]
            peg_rot = self.states["peg_quat"]
            hole_pos = self.states["hole_pos"]
            hole_rot = self.states["hole_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, peg_pos, hole_pos), (eef_rot, peg_rot, hole_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])
	
@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
    vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
    eps (float): Stability value below which small values will be mapped to 0

    Returns:
    tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
    vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
    torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor, Tensor]

    index = 63
    # Compute per-env physical parameters
    target_height = states["hole_size"] + states["peg_size"]
    peg_size = states["peg_size"]
    hole_size = states["hole_size"]


    # distance from hand to the peg
    # reward for lifting peg, in here, I just keep the peg to be above the target height, or it will not walk down
    peg_height = states["peg_grippoint_pos"][:, 2] - reward_settings["table_height"]
    peg_lifted = (peg_height - reward_settings["grippoint_height"]) > 0.02
    lift_reward = peg_lifted
    
    # define the align reward, to make the peg align to the hole both in position and orientation
    d_xy = torch.norm(states["peg_to_hole_pos"][:, :2], dim=-1) 
    d_quat = torch.norm(states["peg_to_hole_quat"], dim=-1)
    align_reward = (1 - torch.tanh(10.0 * (d_quat + d_xy) / 2))
    
    # Define the insert reward
    peg_align_hole = (torch.norm(states["peg_to_hole_quat"], dim=-1) < 0.04)
    peg_close_hole = (torch.norm(states["peg_to_hole_pos"][:, :2], dim=-1) < 0.02)
    d_z = states["peg_to_hole_pos_z"] - 0.085
    dist_reward = (1 - torch.tanh(20 * d_z)) * peg_align_hole * peg_close_hole * peg_lifted

    # define the success standard for insert the peg.
    peg_insert = torch.norm(states["peg_to_hole_pos_z"] - 0.085) < 0.02
    insert_flag = ((states["peg_to_hole_pos_z"] - 0.085) < 0.02) * peg_close_hole

    success = torch.sum(insert_flag)
    if torch.sum(progress_buf) > 150 * int(len(progress_buf)):
        success_rate = int(success) / int(len(insert_flag))
        print("sucess_rate: ", success_rate)

    print("peg_align_hole[index]: ", int(peg_align_hole[index]), "peg_close_hole[index]: ", int(peg_close_hole[index]), "insert_flag", int(insert_flag[index]))
    print("##########################################################")

    stack_reward = peg_insert * peg_align_hole * peg_close_hole

    #TODO:finish the final reward to control the gripper
    # env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
    # if d_z < 0.01:

    # Compose rewards
    rewards = torch.where(
        stack_reward,
        reward_settings["r_stack_scale"] * stack_reward,
        reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_align_scale"] * align_reward,
    )
    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0), torch.ones_like(reset_buf), reset_buf)

    
    return rewards, reset_buf, insert_flag