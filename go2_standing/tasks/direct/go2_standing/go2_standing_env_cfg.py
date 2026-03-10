# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG as GO2_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class Go2StandingEnvCfg(DirectRLEnvCfg):
    # env — 240 Hz physics / 8 decimation = 30 Hz control
    decimation = 8
    episode_length_s = 60.0

    # Go2 has 12 actuated joints: 4 legs x (hip, thigh, calf)
    action_space = 12

    # Observations (50-dim):
    # - joint positions relative to default (12)
    # - joint velocities (12)
    # - base angular velocity (3)
    # - projected gravity (3)
    # - base linear velocity in body frame (3)
    # - base height (1)
    # - previous actions (12)
    # - foot contact binary (4)
    observation_space = 50
    state_space = 0

    # simulation — 240 Hz for accurate contact physics (was 120 Hz)
    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # controlled joints (all 12 leg joints)
    leg_joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    # foot bodies for contact detection (calf links receive foot contact forces)
    foot_body_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

    # default stance joint positions — asymmetric RL-standard pose
    # Hips splayed +/-0.1 for lateral stability, rear thighs more flexed (1.0 vs 0.8)
    default_joint_angles = [
        0.1, 0.8, -1.5,   # FL (hip splayed out)
       -0.1, 0.8, -1.5,   # FR (hip splayed out)
        0.1, 1.0, -1.5,   # RL (rear thigh more flexed)
       -0.1, 1.0, -1.5,   # RR (rear thigh more flexed)
    ]

    # action scale (reduced to prevent toppling with random initial actions)
    action_scale = 0.25

    # Biology-inspired actuator dynamics — EMA low-pass filter
    # alpha=0.4 at 30Hz → time constant ~2.5 steps ≈ 83ms (canine postural reflex timing)
    action_smoothing_alpha = 0.4

    # 1-step action delay to model real actuator communication latency (~30ms)
    action_delay_steps = 1

    # standing height target
    target_base_height = 0.34

    # reward scales — standing task: joint tracking is the primary objective
    rew_scale_orientation = 1.0       # exp(-||proj_grav - [0,0,-1]||^2 / 0.25)
    rew_scale_base_height = 1.0       # exp(-(h - 0.34)^2 / 0.025) — wider Gaussian
    rew_scale_lin_vel = -1.0          # ||lin_vel||^2 — penalize body movement
    rew_scale_ang_vel = -0.5          # ||ang_vel||^2 — penalize body rotation
    rew_scale_joint_pos = 4.0         # exp(-||joint_pos - default||^2 / sigma) — POSITIVE shaped reward, primary driver
    rew_scale_action_rate = -0.01     # ||smoothed_action - prev_smoothed||^2 — mild smoothness penalty
    rew_scale_torque = -0.0001        # ||torque||^2 — mild energy penalty
    rew_scale_alive = 0.1             # per-step alive bonus — small so it doesn't dominate
    rew_scale_termination = -20.0     # falling is catastrophic
    rew_scale_joint_vel = -0.01       # ||joint_vel||^2 — penalize joint movement (standing = still joints)
    rew_scale_foot_symmetry = -0.3    # var(foot_force_fractions) — encourage even weight distribution
    rew_scale_foot_contact = 1.5      # reward per foot in contact — all 4 feet on ground

    # reset conditions
    min_base_height = 0.20
    max_base_height = 0.6

    # external force perturbation — push-then-still pattern
    perturbation_min_quiet_s = 6.0       # min recovery + hold-still time
    perturbation_max_quiet_s = 14.0      # max quiet time (forces patience)
    perturbation_burst_count_min = 1     # min pushes per burst
    perturbation_burst_count_max = 3     # occasional double/triple tap
    perturbation_burst_interval_s = 0.5  # gap between pushes in a burst
    perturbation_force_xy = 70.0         # max horizontal force [N] — increased for robustness
    perturbation_force_z = 35.0          # max vertical force [N]
    perturbation_duration_steps = 5      # sim steps per push
    perturbation_curriculum_steps = 40000

    # domain randomization
    friction_range = [0.5, 1.2]       # uniform friction coefficient
    mass_offset_range = [-1.0, 3.0]   # base link mass offset [kg]

    # observation noise — models real sensor characteristics for sim-to-real transfer
    obs_noise_joint_pos = 0.01        # rad (encoder noise)
    obs_noise_joint_vel = 0.5         # rad/s (encoder differentiation noise)
    obs_noise_ang_vel = 0.1           # rad/s (gyroscope noise)
    obs_noise_lin_vel = 0.1           # m/s (state estimator noise)
    obs_noise_projected_gravity = 0.02  # (IMU orientation noise)
    obs_noise_height = 0.02           # m (height estimator noise)

    # ── Smooth camera follow ──
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.5, 1.5, 0.8),
        lookat=(0.0, 0.0, 0.34),
        origin_type="world",
        env_index=0,
    )
    camera_env_index = 0
    camera_offset = (1.5, 1.0, 0.8)
    camera_lookat_offset = (0.0, 0.0, 0.15)
    camera_smoothing_alpha = 0.05
    camera_height_floor = 0.25
