# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import sample_uniform

from .go2_standing_env_cfg import Go2StandingEnvCfg


class Go2StandingEnv(DirectRLEnv):
    cfg: Go2StandingEnvCfg

    def __init__(self, cfg: Go2StandingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Find joint indices for the controlled joints
        self._leg_joint_ids, _ = self.robot.find_joints(self.cfg.leg_joint_names)

        # Foot body indices (for reference)
        self._foot_body_ids, _ = self.robot.find_bodies(self.cfg.foot_body_names)

        # Default joint positions tensor for standing stance
        self._default_joint_pos = torch.tensor(
            self.cfg.default_joint_angles, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)

        # Previous actions for observation (raw policy outputs)
        self.previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Biology-inspired actuator dynamics: EMA low-pass filter on actions
        self._smoothed_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_smoothed_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._action_alpha = self.cfg.action_smoothing_alpha

        # Action delay buffer — models real actuator communication latency
        self._action_delay_buffer = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # ── Push-then-still perturbation state machine ──
        self._perturbation_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._perturbation_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Control dt for converting seconds → steps
        self._control_dt = self.cfg.sim.dt * self.cfg.decimation

        # Quiet timer: steps remaining before next burst (randomized per env)
        min_quiet_steps = int(self.cfg.perturbation_min_quiet_s / self._control_dt)
        max_quiet_steps = int(self.cfg.perturbation_max_quiet_s / self._control_dt)
        self._quiet_timer = torch.randint(
            min_quiet_steps, max_quiet_steps + 1, (self.num_envs,),
            dtype=torch.long, device=self.device,
        )
        self._min_quiet_steps = min_quiet_steps
        self._max_quiet_steps = max_quiet_steps

        # Burst state: how many pushes remain in current burst, gap counter between pushes
        self._burst_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._burst_gap_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._burst_gap_steps = max(1, int(self.cfg.perturbation_burst_interval_s / self._control_dt))

        # Perturbation curriculum: ramp force from 0 to max over training
        self._global_step_count = 0
        self._perturbation_curriculum_steps = self.cfg.perturbation_curriculum_steps

        # Domain randomization: store default base mass for randomization
        self._default_base_mass = None  # populated on first reset

        # ── Smooth camera follow state ──
        self._cam_smooth_pos: np.ndarray | None = None
        self._cam_alpha = self.cfg.camera_smoothing_alpha
        self._cam_offset = np.array(self.cfg.camera_offset, dtype=np.float64)
        self._cam_lookat_offset = np.array(self.cfg.camera_lookat_offset, dtype=np.float64)
        self._cam_height_floor = self.cfg.camera_height_floor
        self._cam_env_index = self.cfg.camera_env_index

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # Contact sensor for foot contact detection
        contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*_calf",
            update_period=0.0,
            history_length=2,
        )
        self.contact_sensor = ContactSensor(contact_sensor_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(color=(0.5, 0.5, 0.5)))
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation and sensors to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=750.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._global_step_count += 1

        # Action delay: use buffered action from previous step, store current for next
        if self.cfg.action_delay_steps > 0:
            delayed_actions = self._action_delay_buffer.clone()
            self._action_delay_buffer = self.actions.clone()
        else:
            delayed_actions = self.actions

        # EMA smoothing on delayed actions — simulates muscle activation dynamics
        self._prev_smoothed_actions = self._smoothed_actions.clone()
        self._smoothed_actions = (
            self._action_alpha * delayed_actions
            + (1.0 - self._action_alpha) * self._smoothed_actions
        )

        # ── Push-then-still state machine ──
        # Curriculum: ramp force linearly from 0 to max
        curriculum_scale = min(self._global_step_count / max(self._perturbation_curriculum_steps, 1), 1.0)
        effective_force_xy = self.cfg.perturbation_force_xy * curriculum_scale
        effective_force_z = self.cfg.perturbation_force_z * curriculum_scale

        # --- Phase 1: QUIET envs — decrement timer, transition to BURST when expired ---
        in_quiet = (self._burst_remaining == 0) & (self._perturbation_step_counter == 0)
        self._quiet_timer[in_quiet] -= 1

        quiet_expired = in_quiet & (self._quiet_timer <= 0)
        if quiet_expired.any() and effective_force_xy > 0.0:
            ids = quiet_expired.nonzero(as_tuple=False).squeeze(-1)
            n = ids.shape[0]
            self._burst_remaining[ids] = torch.randint(
                self.cfg.perturbation_burst_count_min,
                self.cfg.perturbation_burst_count_max + 1,
                (n,), dtype=torch.long, device=self.device,
            )
            self._burst_gap_timer[ids] = 0

        # --- Phase 2: BURST envs — deliver pushes with gaps ---
        in_burst = (self._burst_remaining > 0) & (self._perturbation_step_counter == 0)
        waiting = in_burst & (self._burst_gap_timer > 0)
        self._burst_gap_timer[waiting] -= 1

        ready_to_push = in_burst & (self._burst_gap_timer <= 0)
        if ready_to_push.any():
            ids = ready_to_push.nonzero(as_tuple=False).squeeze(-1)
            n = ids.shape[0]
            self._perturbation_forces[ids, 0] = sample_uniform(
                -effective_force_xy, effective_force_xy, (n,), self.device
            )
            self._perturbation_forces[ids, 1] = sample_uniform(
                -effective_force_xy, effective_force_xy, (n,), self.device
            )
            self._perturbation_forces[ids, 2] = sample_uniform(
                -effective_force_z, effective_force_z, (n,), self.device
            )
            self._perturbation_step_counter[ids] = self.cfg.perturbation_duration_steps
            self._burst_remaining[ids] -= 1
            self._burst_gap_timer[ids] = self._burst_gap_steps

        # --- Phase 3: Burst complete → sample new quiet duration ---
        burst_done = (self._burst_remaining == 0) & (self._perturbation_step_counter == 0) & ~in_quiet
        if burst_done.any():
            ids = burst_done.nonzero(as_tuple=False).squeeze(-1)
            n = ids.shape[0]
            self._quiet_timer[ids] = torch.randint(
                self._min_quiet_steps, self._max_quiet_steps + 1,
                (n,), dtype=torch.long, device=self.device,
            )

        # --- Apply or clear forces ---
        active = self._perturbation_step_counter > 0
        num_bodies = self.robot.num_bodies
        forces = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
        torques = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
        if active.any():
            forces[:, 0, :] = self._perturbation_forces * active.unsqueeze(-1).float()
            self._perturbation_step_counter[active] -= 1
        self.robot.set_external_force_and_torque(forces, torques)

    def _apply_action(self) -> None:
        # Use smoothed+delayed actions — biological actuators have bandwidth limits
        scaled_actions = self._default_joint_pos + self._smoothed_actions * self.cfg.action_scale
        self.robot.set_joint_position_target(scaled_actions, joint_ids=self._leg_joint_ids)

    def _update_smooth_camera(self) -> None:
        """EMA-smoothed camera that tracks a single robot instance."""
        robot_pos_w = self.robot.data.root_pos_w[self._cam_env_index].cpu().numpy()

        if self._cam_smooth_pos is None:
            self._cam_smooth_pos = robot_pos_w.copy()
        else:
            delta = np.linalg.norm(robot_pos_w - self._cam_smooth_pos)
            if delta > 0.5:
                self._cam_smooth_pos = robot_pos_w.copy()
            else:
                self._cam_smooth_pos += self._cam_alpha * (robot_pos_w - self._cam_smooth_pos)

        anchor = self._cam_smooth_pos.copy()
        anchor[2] = max(anchor[2], self._cam_height_floor)

        eye = anchor + self._cam_offset
        lookat = anchor + self._cam_lookat_offset

        try:
            self.sim.set_camera_view(eye.tolist(), lookat.tolist())
        except (AttributeError, RuntimeError):
            pass

    def _get_observations(self) -> dict:
        self._update_smooth_camera()

        joint_pos = self.robot.data.joint_pos[:, self._leg_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._leg_joint_ids]
        base_ang_vel = self.robot.data.root_ang_vel_b
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_quat = self.robot.data.root_quat_w
        base_height = self.robot.data.root_pos_w[:, 2:3]  # (N, 1)

        # Projected gravity (gravity vector in base frame)
        projected_gravity = quat_rotate_inverse(base_quat, self._gravity_vector)

        # Foot contact detection — binary signal from contact sensor
        foot_contact_forces = self.contact_sensor.data.net_forces_w_history[:, :, -1, 2]  # (N, 4)
        foot_contacts = (foot_contact_forces > 1.0).float()  # binary: 1 if in contact

        # Add observation noise for sim-to-real transfer
        joint_pos_obs = joint_pos - self._default_joint_pos
        joint_vel_obs = joint_vel
        if self.cfg.obs_noise_joint_pos > 0:
            joint_pos_obs = joint_pos_obs + self.cfg.obs_noise_joint_pos * torch.randn_like(joint_pos_obs)
            joint_vel_obs = joint_vel_obs + self.cfg.obs_noise_joint_vel * torch.randn_like(joint_vel_obs)
            base_ang_vel_noisy = base_ang_vel + self.cfg.obs_noise_ang_vel * torch.randn_like(base_ang_vel)
            projected_gravity_noisy = projected_gravity + self.cfg.obs_noise_projected_gravity * torch.randn_like(projected_gravity)
            base_lin_vel_noisy = base_lin_vel + self.cfg.obs_noise_lin_vel * torch.randn_like(base_lin_vel)
            base_height_noisy = base_height + self.cfg.obs_noise_height * torch.randn_like(base_height)
        else:
            base_ang_vel_noisy = base_ang_vel
            projected_gravity_noisy = projected_gravity
            base_lin_vel_noisy = base_lin_vel
            base_height_noisy = base_height

        # 50-dim observation
        obs = torch.cat(
            (
                joint_pos_obs,                  # 12 (relative to default + noise)
                joint_vel_obs,                  # 12
                base_ang_vel_noisy,             # 3
                projected_gravity_noisy,        # 3
                base_lin_vel_noisy,             # 3
                base_height_noisy,              # 1
                self.previous_actions,          # 12 (raw policy outputs, no noise)
                foot_contacts,                  # 4
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        base_pos = self.robot.data.root_pos_w
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        base_quat = self.robot.data.root_quat_w

        projected_gravity = quat_rotate_inverse(base_quat, self._gravity_vector)
        joint_pos = self.robot.data.joint_pos[:, self._leg_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._leg_joint_ids]
        joint_torques = self.robot.data.applied_torque[:, self._leg_joint_ids]

        # Foot contact forces for symmetry reward (from contact sensor)
        foot_contact_forces_z = self.contact_sensor.data.net_forces_w_history[:, :, -1, 2]  # (N, 4)
        foot_contact_forces_z = torch.clamp(foot_contact_forces_z, min=0.0)

        total_reward = compute_standing_rewards(
            # Scales
            self.cfg.rew_scale_orientation,
            self.cfg.rew_scale_base_height,
            self.cfg.rew_scale_lin_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_joint_pos,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_torque,
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_termination,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_foot_symmetry,
            self.cfg.rew_scale_foot_contact,
            # State
            projected_gravity,
            base_pos[:, 2],
            self.cfg.target_base_height,
            base_lin_vel,
            base_ang_vel,
            joint_pos,
            self._default_joint_pos,
            self._smoothed_actions,           # action rate on smoothed (actual actuator commands)
            self._prev_smoothed_actions,      # previous smoothed actions
            joint_torques,
            joint_vel,
            self.reset_terminated,
            foot_contact_forces_z,
        )

        # Update previous actions (raw policy outputs for observation)
        self.previous_actions = self.actions.clone()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_height = self.robot.data.root_pos_w[:, 2]
        base_quat = self.robot.data.root_quat_w
        projected_gravity = quat_rotate_inverse(base_quat, self._gravity_vector)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Terminate if fallen, launched, or tilted > 60 degrees
        out_of_bounds = (base_height < self.cfg.min_base_height) | (base_height > self.cfg.max_base_height)
        out_of_bounds = out_of_bounds | (projected_gravity[:, 2] > -0.5)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset joint positions to default with small random noise
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._leg_joint_ids] += sample_uniform(
            -0.05, 0.05,
            joint_pos[:, self._leg_joint_ids].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # Reset root state
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset action state
        self.previous_actions[env_ids] = 0.0
        self._smoothed_actions[env_ids] = 0.0
        self._prev_smoothed_actions[env_ids] = 0.0
        self._action_delay_buffer[env_ids] = 0.0

        # Reset perturbation state machine
        self._perturbation_forces[env_ids] = 0.0
        self._perturbation_step_counter[env_ids] = 0
        self._burst_remaining[env_ids] = 0
        self._burst_gap_timer[env_ids] = 0
        n_reset = len(env_ids) if not isinstance(env_ids, torch.Tensor) else env_ids.shape[0]
        self._quiet_timer[env_ids] = torch.randint(
            self._min_quiet_steps, self._max_quiet_steps + 1,
            (n_reset,), dtype=torch.long, device=self.device,
        )

        # Snap camera anchor if the tracked env was just reset
        if self._cam_env_index in env_ids:
            self._cam_smooth_pos = None

        # Domain randomization: mass
        # Randomize base link mass on each episode reset for sim-to-real robustness
        if self._default_base_mass is None:
            try:
                masses = self.robot.root_physx_view.get_masses()
                self._default_base_mass = masses[0, 0].item()
            except Exception:
                self._default_base_mass = 0.0  # API not available
        if self._default_base_mass > 0:
            try:
                masses = self.robot.root_physx_view.get_masses().clone()
                mass_offsets = torch.empty(n_reset, device=self.device).uniform_(
                    self.cfg.mass_offset_range[0], self.cfg.mass_offset_range[1]
                )
                masses[env_ids, 0] = self._default_base_mass + mass_offsets
                self.robot.root_physx_view.set_masses(masses, env_ids)
            except Exception:
                pass  # PhysX mass API not available in this version

    @property
    def _gravity_vector(self):
        """Gravity vector in world frame."""
        return torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to a vector."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    return torch.stack([
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx
    ], dim=-1)


@torch.jit.script
def quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion."""
    quat_conj = quat * torch.tensor([1.0, -1.0, -1.0, -1.0], device=quat.device)
    return quat_apply(quat_conj, vec)


@torch.jit.script
def compute_standing_rewards(
    # Scales
    rew_scale_orientation: float,
    rew_scale_base_height: float,
    rew_scale_lin_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_joint_pos: float,
    rew_scale_action_rate: float,
    rew_scale_torque: float,
    rew_scale_alive: float,
    rew_scale_termination: float,
    rew_scale_joint_vel: float,
    rew_scale_foot_symmetry: float,
    rew_scale_foot_contact: float,
    # State
    projected_gravity: torch.Tensor,
    base_height: torch.Tensor,
    target_height: float,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    smoothed_actions: torch.Tensor,
    prev_smoothed_actions: torch.Tensor,
    joint_torques: torch.Tensor,
    joint_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
    foot_contact_forces: torch.Tensor,
) -> torch.Tensor:
    # ── Orientation: exp(-||proj_grav - [0,0,-1]||^2 / 0.25) ──
    # Rewards upright posture. proj_grav=[0,0,-1] means gravity points straight down in body frame.
    gravity_target = torch.tensor([0.0, 0.0, -1.0], device=projected_gravity.device)
    orientation_error = torch.sum(torch.square(projected_gravity - gravity_target), dim=-1)
    rew_orientation = rew_scale_orientation * torch.exp(-orientation_error / 0.25)

    # ── Base height: exp(-(h - target)^2 / 0.025) ──
    # Wider Gaussian (sigma^2=0.025 vs old 0.005) so the reward gradient is useful further from target.
    # At 0.005: 7cm error -> reward=0.37 (too sharp, robot gives up).
    # At 0.025: 7cm error -> reward=0.82 (provides gradient to climb toward target).
    height_error = torch.square(base_height - target_height)
    rew_base_height = rew_scale_base_height * torch.exp(-height_error / 0.025)

    # ── Linear velocity penalty: ||lin_vel||^2 ──
    # Standing robot should be stationary.
    rew_lin_vel = rew_scale_lin_vel * torch.sum(torch.square(base_lin_vel), dim=-1)

    # ── Angular velocity penalty: ||ang_vel||^2 ──
    # Standing robot should not be rotating.
    rew_ang_vel = rew_scale_ang_vel * torch.sum(torch.square(base_ang_vel), dim=-1)

    # ── Joint position tracking: POSITIVE exp reward (THE PRIMARY STANDING REWARD) ──
    # This is the critical fix. The old version used a weak negative penalty (-0.05) with a 0.1rad
    # deadzone, giving the robot almost no incentive to match the default standing pose.
    # Now: positive exponential reward that peaks when joints match default_joint_pos exactly.
    # exp(-sum(error^2) / 0.25) — sigma^2=0.25 gives useful gradient across typical joint range.
    # At scale=4.0, this is the single largest reward term, making joint tracking the dominant objective.
    joint_pos_error_sq = torch.sum(torch.square(joint_pos - default_joint_pos), dim=-1)
    rew_joint_pos = rew_scale_joint_pos * torch.exp(-joint_pos_error_sq / 0.25)

    # ── Joint velocity penalty: ||joint_vel||^2 ──
    # Standing means joints should be still (not oscillating).
    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.square(joint_vel), dim=-1)

    # ── Action rate penalty on smoothed actions ──
    # Penalizes actual actuator jerk, not raw policy noise.
    rew_action_rate = rew_scale_action_rate * torch.sum(
        torch.square(smoothed_actions - prev_smoothed_actions), dim=-1
    )

    # ── Torque penalty: ||torque||^2 ──
    # Mild penalty for energy efficiency — should not dominate over joint tracking.
    rew_torque = rew_scale_torque * torch.sum(torch.square(joint_torques), dim=-1)

    # ── Foot contact reward: reward each foot that is in contact with the ground ──
    # This is the critical missing reward. Without it, the robot gets no bonus for having all
    # 4 feet on the ground. A robot with 1 leg in the air (the observed failure mode) was not
    # penalized because the old symmetry reward only looked at force distribution among feet
    # that were already in contact.
    # Binary: 1 if foot force > 1N, 0 otherwise. Sum gives 0-4, divided by 4 for 0-1 range.
    feet_in_contact = (foot_contact_forces > 1.0).float()  # (N, 4)
    num_feet_in_contact = feet_in_contact.sum(dim=-1)  # (N,) in [0, 4]
    rew_foot_contact = rew_scale_foot_contact * (num_feet_in_contact / 4.0)

    # ── Foot contact symmetry: penalize uneven weight distribution ──
    # Only meaningful when feet are in contact. Encourages equal weight on all 4 feet.
    total_foot_force = foot_contact_forces.sum(dim=-1, keepdim=True).clamp(min=1.0)
    force_fractions = foot_contact_forces / total_foot_force  # (N, 4), sums to ~1
    rew_foot_symmetry = rew_scale_foot_symmetry * torch.var(force_fractions, dim=-1)

    # ── Alive bonus ──
    # Small constant reward for not being terminated. Kept small so it doesn't
    # incentivize survival-only strategies over active standing.
    rew_alive = rew_scale_alive * torch.ones_like(base_height)

    # ── Termination penalty ──
    rew_termination = rew_scale_termination * reset_terminated.float()

    total_reward = (
        rew_orientation
        + rew_base_height
        + rew_lin_vel
        + rew_ang_vel
        + rew_joint_pos
        + rew_joint_vel
        + rew_action_rate
        + rew_torque
        + rew_foot_contact
        + rew_foot_symmetry
        + rew_alive
        + rew_termination
    )

    return total_reward
