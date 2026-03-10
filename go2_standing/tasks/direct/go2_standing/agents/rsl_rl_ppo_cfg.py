# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64   # longer rollouts capture full push-recovery cycles
    max_iterations = 5000
    save_interval = 200
    experiment_name = "go2_standing"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,     # was 2.0 — too high causes wild initial actions that get penalized,
                                # teaching the robot to output zeros (do-nothing policy)
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,      # was 0.05 — standing is NOT a multi-modal task, excessive entropy
                                # prevents convergence to the single correct pose
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,   # faster early learning (adaptive schedule throttles later)
        schedule="adaptive",
        gamma=0.99,             # was 0.995 — shorter horizon appropriate for standing (not locomotion)
        lam=0.95,
        desired_kl=0.01,        # was 0.015 — tighter KL for more stable convergence
        max_grad_norm=1.0,
    )
