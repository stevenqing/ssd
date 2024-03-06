import time
from collections import deque
from typing import Any, Dict, List, Optional, Type, Union
import torch.nn.functional as F
import gym
import numpy as np
import torch as th
import wandb
from gym.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import (configure_logger, obs_as_tensor,
                                            safe_mean)
from stable_baselines3.common.vec_env import DummyVecEnv


class DummyGymEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class IndependentPPO(OnPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        num_agents: int,
        env: GymEnv,
        learning_rate: Union[float, Schedule] = 1e-4,
        n_steps: int = 1000,
        batch_size: int = 6000,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 40,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        alpha: float = 0.55,
        beta: float = 0.55,
        model: str = 'baseline',
        using_reward_timestep: int = 2000000,
        enable_trajs_learning: bool = False,
        env_name: str = 'harvest',
        use_collective_reward: bool = False,
        inequity_averse_reward: bool = False,
    ):
        self.env = env
        self.env_name = env_name
        self.num_agents = num_agents
        self.num_envs = env.num_envs // num_agents
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_steps = n_steps
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self._logger = None
        self.alpha = alpha
        self.beta = beta
        self.model = model
        self.using_reward_timestep = using_reward_timestep
        self.enable_trajs_learning = enable_trajs_learning
        self.use_collective_reward = use_collective_reward
        self.inequity_averse_reward = inequity_averse_reward
        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        dummy_env = DummyVecEnv([env_fn] * self.num_envs)
        self.policies = [
            PPO(
                policy=policy,
                env=dummy_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                clip_range_vf=clip_range_vf,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                device=device,
                num_agents=self.num_agents,
                model=self.model,
                enable_trajs_learning=self.enable_trajs_learning,
                polid=polid,
                enable_reward_model_learning=self.using_reward_timestep,
            )
            for polid in range(self.num_agents)
        ]
        self.previous_all_last_obs_traj = None
        self.previous_all_actions_traj = None
        self.previous_all_rewards_traj = None
        self.prev_latent_state = None
    def learn(
        self,
        total_timesteps: int,
        callbacks: Optional[List[MaybeCallback]] = None,
        log_interval: int = 1,
        tb_log_name: str = "IndependentPPO",
        reset_num_timesteps: bool = True,
    ):

        num_timesteps = 0
        all_total_timesteps = []
        if not callbacks:
            callbacks = [None] * self.num_agents
        self._logger = configure_logger(
            self.verbose,
            self.tensorboard_log,
            tb_log_name,
            reset_num_timesteps,
        )
        logdir = self.logger.dir

        # Setup for each policy
        for polid, policy in enumerate(self.policies):
            policy.start_time = time.time()
            if policy.ep_info_buffer is None or reset_num_timesteps:
                policy.ep_info_buffer = deque(maxlen=100)
                policy.ep_success_buffer = deque(maxlen=100)

            if policy.action_noise is not None:
                policy.action_noise.reset()

            if reset_num_timesteps:
                policy.num_timesteps = 0
                policy._episode_num = 0
                all_total_timesteps.append(total_timesteps)
                policy._total_timesteps = total_timesteps
            else:
                # make sure training timestamps are ahead of internal counter
                all_total_timesteps.append(total_timesteps + policy.num_timesteps)
                policy._total_timesteps = total_timesteps + policy.num_timesteps

            policy._logger = configure_logger(
                policy.verbose,
                logdir,
                "policy",
                reset_num_timesteps,
            )

            callbacks[polid] = policy._init_callback(callbacks[polid])

        for callback in callbacks:
            callback.on_training_start(locals(), globals())

        last_obs = self.env.reset()
        for policy in self.policies:
            policy._last_episode_starts = np.ones((self.num_envs,), dtype=bool)

        while num_timesteps < total_timesteps:
            if self.enable_trajs_learning or self.model == 'vae':
                last_obs = self.collect_trajs_rollouts(last_obs, callbacks,num_timesteps)
            else:
                last_obs = self.collect_rollouts(last_obs, callbacks,num_timesteps)
            num_timesteps += self.num_envs * self.n_steps
            SW_ep_rew_mean = 0
            for polid, policy in enumerate(self.policies):
                policy._update_current_progress_remaining(
                    policy.num_timesteps, total_timesteps
                )
                if log_interval is not None and num_timesteps % log_interval == 0:
                    fps = int(policy.num_timesteps / (time.time() - policy.start_time))
                    wandb.log({f"{polid}/fps": fps}, step=num_timesteps)
                    if self.use_collective_reward:
                        wandb.log({f"{polid}/ep_rew_mean": safe_mean([ep_info["r"] for ep_info in policy.ep_info_buffer])/self.num_agents}, step=num_timesteps)
                    else:
                        wandb.log({f"{polid}/ep_rew_mean": safe_mean([ep_info["r"] for ep_info in policy.ep_info_buffer])}, step=num_timesteps)
                    wandb.log({f"{polid}/ep_len_mean": policy.ep_info_buffer[-1]["l"]}, step=num_timesteps)
                    wandb.log({f"{polid}/time_elapsed": int(time.time() - policy.start_time)}, step=num_timesteps)
                    wandb.log({f"{polid}/total_timesteps": policy.num_timesteps}, step=num_timesteps)
                    
                    SW_ep_rew_mean += safe_mean([ep_info["r"] for ep_info in policy.ep_info_buffer])
                    ep_cf_reward = np.sum(policy.rollout_buffer.cf_rewards)
                    wandb.log({f"{polid}/cf_reward": ep_cf_reward}, step=num_timesteps)

                    if self.env_name == 'harvest':
                        wandb.log({f"{polid}/zap_behavior": (policy.rollout_buffer.actions==7).sum()}, step=num_timesteps)
                    elif self.env_name == 'cleanup':
                        wandb.log({f"{polid}/zap_behavior": (policy.rollout_buffer.actions==7).sum()}, step=num_timesteps)
                        wandb.log({f"{polid}/clean_behavior": (policy.rollout_buffer.actions==8).sum()}, step=num_timesteps)

                    policy.logger.record("policy_id", polid, exclude="tensorboard")
                    policy.logger.record(
                        "time/iterations", num_timesteps, exclude="tensorboard"
                    )
                    if (
                        len(policy.ep_info_buffer) > 0
                        and len(policy.ep_info_buffer[0]) > 0
                    ):
                        policy.logger.record(
                            "rollout/ep_rew_mean",
                            safe_mean(
                                [ep_info["r"] for ep_info in policy.ep_info_buffer]
                            ),
                        )
                        policy.logger.record(
                            "rollout/ep_len_mean",
                            safe_mean(
                                [ep_info["l"] for ep_info in policy.ep_info_buffer]
                            ),
                        )
                    policy.logger.record("time/fps", fps)
                    policy.logger.record(
                        "time/time_elapsed",
                        int(time.time() - policy.start_time),
                        exclude="tensorboard",
                    )
                    policy.logger.record(
                        "time/total_timesteps",
                        policy.num_timesteps,
                        exclude="tensorboard",
                    )
                    policy.logger.dump(step=policy.num_timesteps)

                policy.train()
            if self.use_collective_reward:
                wandb.log({"SW_ep_rew_mean": SW_ep_rew_mean/(self.num_agents * 2)}, step=num_timesteps)
                wandb.log({"SW_ep_rew_total": SW_ep_rew_mean/self.num_agents}, step=num_timesteps)
            else:
                wandb.log({"SW_ep_rew_mean": SW_ep_rew_mean/self.num_agents}, step=num_timesteps)
                wandb.log({"SW_ep_rew_total": SW_ep_rew_mean}, step=num_timesteps)
        for callback in callbacks:
            callback.on_training_end()

    def collect_trajs_rollouts(self, last_obs, callbacks,num_timesteps):

        all_last_episode_starts = [None] * self.num_agents
        all_obs = [None] * self.num_agents
        all_last_obs = [None] * self.num_agents
        all_rewards = [None] * self.num_agents
        all_dones = [None] * self.num_agents
        all_infos = [None] * self.num_agents
        all_distributions = [None] * self.num_agents

        all_obs_trajs,all_actions_trajs,all_rewards_trajs = [],[],[]
        steps = 0

        for polid, policy in enumerate(self.policies):
            for envid in range(self.num_envs):
                assert (
                    last_obs[envid * self.num_agents + polid] is not None
                ), f"No previous observation was provided for env_{envid}_policy_{polid}"
            all_last_obs[polid] = np.array(
                [
                    last_obs[envid * self.num_agents + polid]
                    for envid in range(self.num_envs)
                ]
            )
            policy.policy.set_training_mode(False)
            policy.rollout_buffer.agent_number = self.num_agents
            policy.rollout_buffer.model = self.model
            policy.rollout_buffer.reset()
            callbacks[polid].on_rollout_start()
            all_last_episode_starts[polid] = policy._last_episode_starts

        while steps < self.n_steps:
            all_actions = [None] * self.num_agents
            all_values = [None] * self.num_agents
            all_log_probs = [None] * self.num_agents
            all_clipped_actions = [None] * self.num_agents
            with th.no_grad():
                for polid, policy in enumerate(self.policies):
                    obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                    (
                        all_actions[polid],
                        all_values[polid],
                        all_log_probs[polid],
                        all_distributions[polid],
                    ) = policy.policy.forward(obs_tensor)
                    clipped_actions = all_actions[polid].cpu().numpy()
                    if isinstance(self.action_space, Box):
                        clipped_actions = np.clip(
                            clipped_actions,
                            self.action_space.low,
                            self.action_space.high,
                        )
                    elif isinstance(self.action_space, Discrete):
                        # get integer from numpy array
                        clipped_actions = np.array(
                            [action.item() for action in clipped_actions]
                        )
                    all_clipped_actions[polid] = clipped_actions

            all_clipped_actions = (
                np.vstack(all_clipped_actions).transpose().reshape(-1)
            )  # reshape as (env, action)
            obs, rewards, dones, infos = self.env.step(all_clipped_actions)
            
            if dones.any():
                all_obs_trajs.append(all_last_obs)
                all_actions_trajs.append(np.array([action.cpu().numpy() for action in all_actions]))
                for polid in range(self.num_agents):
                    all_rewards[polid] = np.array(
                    [
                        rewards[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_rewards_trajs.append(all_rewards)
                if self.previous_all_last_obs_traj is None or self.previous_all_actions_traj is None or self.previous_all_rewards_traj is None:
                    self.previous_all_last_obs_traj = np.array(all_obs_trajs)
                    self.previous_all_actions_traj = np.array(all_actions_trajs)
                    self.previous_all_rewards_traj = np.array(all_rewards_trajs) 
                all_obs_trajs,all_actions_trajs,all_rewards_trajs = np.array(all_obs_trajs),np.array(all_actions_trajs),np.array(all_rewards_trajs)
            else:
                all_obs_trajs.append(all_last_obs)
                all_actions_trajs.append(np.array([action.cpu().numpy() for action in all_actions]))
                for polid in range(self.num_agents):
                    all_rewards[polid] = np.array(
                    [
                        rewards[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_rewards_trajs.append(all_rewards)
                all_rewards = [None] * self.num_agents
            ############################################
            for polid in range(self.num_agents):
                all_obs[polid] = np.array(
                    [
                        obs[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_rewards[polid] = np.array(
                    [
                        rewards[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_dones[polid] = np.array(
                    [
                        dones[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_infos[polid] = np.array(
                    [
                        infos[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )

            for policy in self.policies:
                policy.num_timesteps += self.num_envs

            for callback in callbacks:
                callback.update_locals(locals())
            if not [callback.on_step() for callback in callbacks]:
                break

            for polid, policy in enumerate(self.policies):
                policy._update_info_buffer(all_infos[polid])

            steps += 1


            # add data to the rollout buffers
            for polid, policy in enumerate(self.policies):
                if isinstance(self.action_space, Discrete):
                    all_actions[polid] = all_actions[polid].reshape(-1, 1)
                all_actions[polid] = all_actions[polid].cpu().numpy()
            rollout_all_actions = all_actions
            for polid, policy in enumerate(self.policies):
                if self.model == 'baseline':
                    policy.rollout_buffer.add(
                        all_last_obs[polid],
                        all_actions[polid],
                        all_rewards[polid],
                        all_last_episode_starts[polid],
                        all_values[polid],
                        all_log_probs[polid],
                    )
                elif self.model == 'vae':
                    cf_rewards = self.compute_transition_cf_rewards(policy,all_last_obs,all_rewards,all_actions,polid,all_distributions) #SPEED
                    policy.rollout_buffer.add_sw_traj(
                        all_last_obs[polid],
                        all_actions[polid],
                        all_rewards[polid],
                        all_last_episode_starts[polid],
                        all_values[polid],
                        all_log_probs[polid],
                        all_last_obs,
                        rollout_all_actions,
                        all_rewards,
                        cf_rewards,
                        all_obs_trajs,
                        all_actions_trajs,
                        all_rewards_trajs,
                        self.previous_all_last_obs_traj,
                        self.previous_all_actions_traj,
                        self.previous_all_rewards_traj,
                    )
                else:
                    if self.model == 'team':
                        policy.rollout_buffer.add_sw(
                            all_last_obs[polid],
                            all_actions[polid],
                            all_rewards[polid],
                            all_last_episode_starts[polid],
                            all_values[polid],
                            all_log_probs[polid],
                            all_last_obs,
                            rollout_all_actions,
                            all_rewards,
                            cf_rewards=None,
                        )
                    else:
                        cf_rewards = self.compute_cf_rewards(policy,all_last_obs,all_actions,polid) 
                        if num_timesteps <= self.using_reward_timestep:
                            cf_rewards = np.zeros_like(cf_rewards)
                        if self.enable_trajs_learning:
                            policy.rollout_buffer.add_sw_traj( # add_sw
                                all_last_obs[polid],
                                all_actions[polid],
                                all_rewards[polid],
                                all_last_episode_starts[polid],
                                all_values[polid],
                                all_log_probs[polid],
                                all_last_obs,
                                rollout_all_actions,
                                all_rewards,
                                cf_rewards,
                                all_obs_trajs,
                                all_actions_trajs,
                                all_rewards_trajs,
                                self.previous_all_last_obs_traj,
                                self.previous_all_actions_traj,
                                self.previous_all_rewards_traj,
                            )
                        else:   
                            policy.rollout_buffer.add_sw(
                                all_last_obs[polid],
                                all_actions[polid],
                                all_rewards[polid],
                                all_last_episode_starts[polid],
                                all_values[polid],
                                all_log_probs[polid],
                                all_last_obs,
                                rollout_all_actions,
                                all_rewards,
                                cf_rewards,
                            )
            if isinstance(all_obs_trajs, np.ndarray):
                self.previous_all_last_obs_traj = all_obs_trajs
                self.previous_all_actions_traj = all_actions_trajs
                self.previous_all_rewards_traj = all_rewards_trajs
                all_obs_trajs,all_actions_trajs,all_rewards_trajs = [],[],[]

            all_last_obs = all_obs
            all_last_episode_starts = all_dones

        with th.no_grad():
            for polid, policy in enumerate(self.policies):
                obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                _, value, _ = policy.policy.forward(obs_tensor)
                if self.model == 'baseline':
                        policy.rollout_buffer.compute_returns_and_advantage(
                        last_values=value, dones=all_dones[polid]
                    )
                else:
                    if self.model == 'team':
                        policy.rollout_buffer.compute_sw_returns_and_advantage(
                        last_values=value, dones=all_dones[polid], alpha=self.alpha, use_team_reward=True
                    )
                    else:
                        policy.rollout_buffer.compute_sw_returns_and_advantage(
                        last_values=value, dones=all_dones[polid], alpha=self.alpha
                    )

        for callback in callbacks:
            callback.on_rollout_end()

        for polid, policy in enumerate(self.policies):
            policy._last_episode_starts = all_last_episode_starts[polid]

        return obs



    def collect_rollouts(self, last_obs, callbacks,num_timesteps):

        all_last_episode_starts = [None] * self.num_agents
        all_obs = [None] * self.num_agents
        all_last_obs = [None] * self.num_agents
        all_rewards = [None] * self.num_agents
        all_dones = [None] * self.num_agents
        all_infos = [None] * self.num_agents
        all_distributions = [None] * self.num_agents
        steps = 0

        for polid, policy in enumerate(self.policies):
            for envid in range(self.num_envs):
                assert (
                    last_obs[envid * self.num_agents + polid] is not None
                ), f"No previous observation was provided for env_{envid}_policy_{polid}"
            all_last_obs[polid] = np.array(
                [
                    last_obs[envid * self.num_agents + polid]
                    for envid in range(self.num_envs)
                ]
            )
            policy.policy.set_training_mode(False)
            policy.rollout_buffer.agent_number = self.num_agents
            policy.rollout_buffer.model = self.model
            policy.rollout_buffer.reset()
            callbacks[polid].on_rollout_start()
            all_last_episode_starts[polid] = policy._last_episode_starts

        while steps < self.n_steps:
            all_actions = [None] * self.num_agents
            all_values = [None] * self.num_agents
            all_log_probs = [None] * self.num_agents
            all_clipped_actions = [None] * self.num_agents
            with th.no_grad():
                for polid, policy in enumerate(self.policies):
                    obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                    (
                        all_actions[polid],
                        all_values[polid],
                        all_log_probs[polid],
                        all_distributions[polid],
                    ) = policy.policy.forward(obs_tensor)
                    clipped_actions = all_actions[polid].cpu().numpy()
                    if isinstance(self.action_space, Box):
                        clipped_actions = np.clip(
                            clipped_actions,
                            self.action_space.low,
                            self.action_space.high,
                        )
                    elif isinstance(self.action_space, Discrete):
                        # get integer from numpy array
                        clipped_actions = np.array(
                            [action.item() for action in clipped_actions]
                        )
                    all_clipped_actions[polid] = clipped_actions

            all_clipped_actions = (
                np.vstack(all_clipped_actions).transpose().reshape(-1)
            )  # reshape as (env, action)
            obs, rewards, dones, infos = self.env.step(all_clipped_actions)

            for polid in range(self.num_agents):
                all_obs[polid] = np.array(
                    [
                        obs[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_rewards[polid] = np.array(
                    [
                        rewards[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_dones[polid] = np.array(
                    [
                        dones[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_infos[polid] = np.array(
                    [
                        infos[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )

            for policy in self.policies:
                policy.num_timesteps += self.num_envs

            for callback in callbacks:
                callback.update_locals(locals())
            if not [callback.on_step() for callback in callbacks]:
                break

            for polid, policy in enumerate(self.policies):
                policy._update_info_buffer(all_infos[polid])

            steps += 1

            # add data to the rollout buffers
            for polid, policy in enumerate(self.policies):
                if isinstance(self.action_space, Discrete):
                    all_actions[polid] = all_actions[polid].reshape(-1, 1)
                all_actions[polid] = all_actions[polid].cpu().numpy()
            rollout_all_actions = all_actions
            for polid, policy in enumerate(self.policies):
                if self.model == 'baseline':
                    if self.inequity_averse_reward:
                        inequity_reward = self.compute_inequity_averse_rewards(all_rewards[polid],all_rewards) #SPEED
                        policy.rollout_buffer.add_inequity_sw(
                            all_last_obs[polid],
                            all_actions[polid],
                            all_rewards[polid],
                            all_last_episode_starts[polid],
                            all_values[polid],
                            all_log_probs[polid],
                            all_last_obs,
                            rollout_all_actions,
                            all_rewards,
                            inequity_reward,
                        )
                    else:
                        policy.rollout_buffer.add(
                            all_last_obs[polid],
                            all_actions[polid],
                            all_rewards[polid],
                            all_last_episode_starts[polid],
                            all_values[polid],
                            all_log_probs[polid],
                        )

                else:
                    if self.model == 'team':
                        policy.rollout_buffer.add_sw(
                            all_last_obs[polid],
                            all_actions[polid],
                            all_rewards[polid],
                            all_last_episode_starts[polid],
                            all_values[polid],
                            all_log_probs[polid],
                            all_last_obs,
                            rollout_all_actions,
                            all_rewards,
                            cf_rewards=None,
                        )
                    elif self.model == 'vae':
                        cf_rewards = self.compute_transition_cf_rewards(policy,all_last_obs,all_rewards,all_actions,polid,all_distributions) #SPEED
                        policy.rollout_buffer.add_sw(
                            all_last_obs[polid],
                            all_actions[polid],
                            all_rewards[polid],
                            all_last_episode_starts[polid],
                            all_values[polid],
                            all_log_probs[polid],
                            all_last_obs,
                            rollout_all_actions,
                            all_rewards,
                            cf_rewards,
                        )
                    else:
                        cf_rewards = self.compute_cf_rewards(policy,all_last_obs,all_actions,polid,all_distributions) #SPEED
                        policy.rollout_buffer.add_sw(
                            all_last_obs[polid],
                            all_actions[polid],
                            all_rewards[polid],
                            all_last_episode_starts[polid],
                            all_values[polid],
                            all_log_probs[polid],
                            all_last_obs,
                            rollout_all_actions,
                            all_rewards,
                            cf_rewards,
                        )
            all_last_obs = all_obs
            all_last_episode_starts = all_dones

        with th.no_grad():
            for polid, policy in enumerate(self.policies):
                obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                _, value, _,_ = policy.policy.forward(obs_tensor)
                if self.model == 'baseline':
                    if self.inequity_averse_reward:
                        policy.rollout_buffer.compute_inequity_returns_and_advantage(
                        last_values=value, dones=all_dones[polid], polid=polid
                    )
                    else:
                        policy.rollout_buffer.compute_returns_and_advantage(
                        last_values=value, dones=all_dones[polid]
                    )
                else:
                    if self.model == 'team':
                        policy.rollout_buffer.compute_sw_returns_and_advantage(
                        last_values=value, dones=all_dones[polid], alpha=self.alpha, use_team_reward=True
                    )
                    else:
                        policy.rollout_buffer.compute_sw_returns_and_advantage(
                        last_values=value, dones=all_dones[polid], alpha=self.alpha
                    )

        for callback in callbacks:
            callback.on_rollout_end()

        for polid, policy in enumerate(self.policies):
            policy._last_episode_starts = all_last_episode_starts[polid]

        return obs

    def compute_inequity_averse_rewards(self,individual_reward,all_rewards):
        diff = np.array([individual_reward - reward for reward in all_rewards])
        dis_inequity = self.alpha * sum(diff[diff > 0])
        adv_inequity = self.beta * sum(diff[diff < 0])
        inequity_reward = individual_reward - (dis_inequity + adv_inequity) / (self.num_agents -1)
        return inequity_reward


    def compute_transition_cf_rewards(self,policy,all_last_obs,all_rewards,all_actions,polid,all_distributions,sample_number=10):
        all_cf_rewards = []

        all_last_obs = obs_as_tensor(np.array(all_last_obs), policy.device)
        all_rewards = obs_as_tensor(np.transpose(np.array(all_rewards),(1,0)), policy.device)
        all_actions = obs_as_tensor(np.transpose(np.array(all_actions),(1,0,2)), policy.device)

        all_obs_features = []
        for i in range(self.num_agents):
            all_obs_features.append(policy.policy.extract_features(all_last_obs[i]))
        all_obs_features = th.stack(all_obs_features,dim=0).permute(1,0,2)
        all_obs_features = all_obs_features.reshape(all_obs_features.shape[0],-1)

        all_actions_one_hot = all_actions[:,polid,:]
        eye_matrix = th.eye(self.action_space.n,device=all_actions_one_hot.device)
        all_actions_one_hot = eye_matrix[all_actions_one_hot]
        all_actions_one_hot = all_actions_one_hot.unsqueeze(1)
        all_actions_one_hot = all_actions_one_hot.repeat(1,1,sample_number,1)
        all_actions_one_hot = all_actions_one_hot.repeat(1,self.num_agents,1,1)
        all_actions_one_hot_list = all_actions_one_hot.permute(1,0,2,3)

        total_actions = [None] * self.num_agents
        for i in range(self.num_agents):
            if i != polid:
                actions_one_hot_copy = all_actions_one_hot_list.clone()
                cf_action_i = self.generate_samples(all_distributions[i],sample_number)
                cf_action_i = cf_action_i.permute(1,0,2,3).squeeze(0)
                actions_one_hot_copy[i,:,:,:] = cf_action_i

                total_actions[i] = actions_one_hot_copy
        total_cf_rewards = []
        for all_actions_one_hot in total_actions:
            if all_actions_one_hot is not None:
                all_actions_one_hot = all_actions_one_hot.permute(1,2,0,3)

                all_actions_one_hot = all_actions_one_hot.reshape(all_actions_one_hot.shape[0],all_actions_one_hot.shape[1],-1).permute(1,0,2)
                all_actions_one_hot_flatten = all_actions_one_hot.reshape(-1,all_actions_one_hot.shape[-1])
                all_obs_features_copy = all_obs_features.clone().repeat(all_actions_one_hot.shape[0],1,1)
                
                all_obs_actions_features = th.cat((all_obs_features_copy,all_actions_one_hot),dim=-1) # Here should be (num_env, sample_number, (feature_dim+action_dim)*num_agents). We won't permute it since we will be using the same form on the action_one_hot(for the transition model)
                all_obs_actions_features = all_obs_actions_features.reshape(-1,all_obs_actions_features.shape[-1])

                # Let the dimension of the rewards be (num_env, sample_number, num_agents)
                all_rewards_copy = all_rewards.clone().unsqueeze(1)
                all_rewards_copy = all_rewards_copy.repeat(1,sample_number,1)
                all_rewards_copy = all_rewards_copy.reshape(-1,all_rewards_copy.shape[-1])

                if self.prev_latent_state == None:
                    prev_latent_state_size = (all_obs_actions_features.shape[0],int((all_obs_actions_features.shape[-1]-self.num_agents*self.action_space.n)/self.num_agents))
                    self.prev_latent_state = th.randn(prev_latent_state_size)
                else:
                    self.prev_latent_state = self.prev_latent_state.squeeze(1)

                # prev_latent_state = policy.policy.vae_net.encoder(self.prev_latent_state.to(all_obs_actions_features.device), all_obs_actions_features, all_rewards_copy).rsample()
                prev_latent_state = policy.policy.vae_net.encoder(all_obs_actions_features, all_rewards_copy).rsample()
                latent_state = policy.policy.transition_net(prev_latent_state.squeeze(1), all_actions_one_hot_flatten)
                
                latent_state_action = th.cat((latent_state,all_actions_one_hot_flatten),dim=-1)
                all_cf_rewards = policy.policy.reward_net(latent_state_action)[0].squeeze().reshape(self.num_envs,-1,self.num_agents)

                all_cf_rewards = th.mean(all_cf_rewards,dim=1) #SPEED? Not sure in here
                total_cf_rewards.append(all_cf_rewards)
        total_cf_rewards = th.stack(total_cf_rewards,dim=0)
        total_cf_rewards = th.mean(total_cf_rewards,dim=0).cpu().detach().numpy()        
        self.prev_latent_state = prev_latent_state
        return total_cf_rewards
    

    def compute_cf_rewards(self,policy,all_last_obs,all_actions,polid,all_distributions,sample_number=10):
        all_cf_rewards = []

        all_last_obs = obs_as_tensor(np.array(all_last_obs), policy.device)
        all_actions = obs_as_tensor(np.transpose(np.array(all_actions),(1,0,2)), policy.device)
        
        # extract obs features
        all_obs_features = []
        for i in range(self.num_agents):
            all_obs_features.append(policy.policy.extract_features(all_last_obs[i]))
        all_obs_features = th.stack(all_obs_features,dim=0).permute(1,0,2)
        all_obs_features = all_obs_features.reshape(all_obs_features.shape[0],-1)

        all_actions_one_hot = all_actions[:,polid,:]
        eye_matrix = th.eye(self.action_space.n,device=all_actions_one_hot.device)
        all_actions_one_hot = eye_matrix[all_actions_one_hot]
        all_actions_one_hot = all_actions_one_hot.unsqueeze(1)
        all_actions_one_hot = all_actions_one_hot.repeat(1,1,sample_number,1)
        all_actions_one_hot = all_actions_one_hot.repeat(1,self.num_agents,1,1)
        all_actions_one_hot_list = all_actions_one_hot.permute(1,0,2,3)

        # for i in range(self.num_agents):
        #     if i != polid:
        #         cf_action_i = self.generate_samples(all_distributions[i],sample_number)
        #         cf_action_i = cf_action_i.permute(1,0,2,3).squeeze(0)
        #         all_actions_one_hot[i,:,:,:] = cf_action_i
        total_actions = [None] * self.num_agents
        for i in range(self.num_agents):
            if i != polid:
                actions_one_hot_copy = all_actions_one_hot_list.clone()
                cf_action_i = self.generate_samples(all_distributions[i],sample_number)
                cf_action_i = cf_action_i.permute(1,0,2,3).squeeze(0)
                actions_one_hot_copy[i,:,:,:] = cf_action_i

                total_actions[i] = actions_one_hot_copy
        total_cf_rewards = []
        for all_actions_one_hot in total_actions:
            if all_actions_one_hot is not None:
                all_actions_one_hot = all_actions_one_hot.permute(1,2,0,3)

                all_actions_one_hot = all_actions_one_hot.reshape(all_actions_one_hot.shape[0],all_actions_one_hot.shape[1],-1).permute(1,0,2)
                all_obs_features_copy = all_obs_features.clone().repeat(all_actions_one_hot.shape[0],1,1)
                
                all_obs_actions_features = th.cat((all_obs_features_copy,all_actions_one_hot),dim=-1).permute(1,0,2)
                all_obs_actions_features = all_obs_actions_features.reshape(-1,all_obs_actions_features.shape[-1])
                
                all_cf_rewards = policy.policy.reward_net(all_obs_actions_features,self.num_agents)[0].squeeze().reshape(self.num_envs,-1,self.num_agents)

                all_cf_rewards = th.mean(all_cf_rewards,dim=1) #SPEED? Not sure in here
                total_cf_rewards.append(all_cf_rewards)
        total_cf_rewards = th.stack(total_cf_rewards,dim=0)
        total_cf_rewards = th.mean(total_cf_rewards,dim=0).cpu().detach().numpy()        
        return total_cf_rewards

    def generate_samples(self,distribution,sample_number):
        all_samples = []
        for i in range(sample_number): #SPEED, can sample method apply to the whole tensor?
            all_samples.append(distribution.sample())
        all_samples = th.stack(all_samples,dim=0)
        eye_matrix = th.eye(self.action_space.n,device=all_samples.device)
        all_samples = eye_matrix[all_samples]
        all_samples = all_samples.permute(1,0,2)
        all_samples = all_samples.unsqueeze(1)
        return all_samples


    # def compute_cf_rewards(self,policy,all_last_obs,all_actions,polid,all_distributions):
    #     all_cf_rewards = []

    #     all_last_obs = obs_as_tensor(np.array(all_last_obs), policy.device)
    #     all_actions = obs_as_tensor(np.transpose(np.array(all_actions),(1,0,2)), policy.device)
        
    #     # extract obs features
    #     all_obs_features = []
    #     for i in range(self.num_agents):
    #         all_obs_features.append(policy.policy.extract_features(all_last_obs[i]))
    #     all_obs_features = th.stack(all_obs_features,dim=0).permute(1,0,2)
    #     all_obs_features = all_obs_features.reshape(all_obs_features.shape[0],-1)
    #     index = 0
    #     all_actions_one_hot = F.one_hot(all_actions, num_classes=self.action_space.n).repeat(1,1,(self.num_agents-1) * self.action_space.n,1)
    #     for i in range(self.num_agents):
    #         if i != polid:
    #             all_actions_i = th.eye(self.action_space.n).repeat(self.num_envs,1,1)
    #             all_actions_one_hot[:,i,index:index+self.action_space.n,:] = all_actions_i
    #             index += self.action_space.n
    #     # Need to double check here, to see if the cf is correct, (num_envs, num_agents, num_cf, num_action_space)
    #     all_actions_one_hot = all_actions_one_hot.permute(0,2,1,3)
    #     all_actions_one_hot = all_actions_one_hot.reshape(all_actions_one_hot.shape[0],all_actions_one_hot.shape[1],-1).permute(1,0,2)
    #     all_obs_features = all_obs_features.repeat(all_actions_one_hot.shape[0],1,1)
        
    #     all_obs_actions_features = th.cat((all_obs_features,all_actions_one_hot),dim=-1).permute(1,0,2)
    #     all_obs_actions_features = all_obs_actions_features.reshape(-1,all_obs_actions_features.shape[-1])
        
    #     all_cf_rewards = policy.policy.reward_net(all_obs_actions_features,self.num_agents)[0].squeeze().reshape(self.num_envs,-1,self.num_agents)
    #     all_cf_rewards = th.mean(all_cf_rewards,dim=1).cpu().detach().numpy()
    #     return all_cf_rewards


    @classmethod
    def load(
        cls,
        path: str,
        policy: Union[str, Type[ActorCriticPolicy]],
        num_agents: int,
        env: GymEnv,
        n_steps: int,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        **kwargs,
    ) -> "IndependentPPO":
        model = cls(
            policy=policy,
            num_agents=num_agents,
            env=env,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **kwargs,
        )
        env_fn = lambda: DummyGymEnv(env.observation_space, env.action_space)
        dummy_env = DummyVecEnv([env_fn] * (env.num_envs // num_agents))
        for polid in range(num_agents):
            model.policies[polid] = PPO.load(
                path=path + f"/policy_{polid + 1}/model", env=dummy_env, **kwargs
            )
        return model

    def save(self, path: str) -> None:
        for polid in range(self.num_agents):
            self.policies[polid].save(path=path + f"/policy_{polid + 1}/model")
