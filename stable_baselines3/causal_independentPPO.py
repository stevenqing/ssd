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


class Causal_IndependentPPO(OnPolicyAlgorithm):
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
        alpha: float = 0.5,
        model: str = 'baseline',
    ):
        self.env = env
        self.num_agents = num_agents
        self.num_envs = env.num_envs // num_agents
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_steps = n_steps
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self._logger = None
        self.alpha = alpha
        self.model = model
        self.prev_state = None
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
            )
            for _ in range(self.num_agents)
        ]

    def learn(
        self,
        total_timesteps: int,
        callbacks: Optional[List[MaybeCallback]] = None,
        log_interval: int = 1,
        tb_log_name: str = "Causal_IndependentPPO",
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
            last_obs = self.collect_rollouts(last_obs, callbacks, num_timesteps)
            
            num_timesteps += self.num_envs * self.n_steps
            SW_ep_rew_mean = 0
            
            for polid, policy in enumerate(self.policies):
                policy._update_current_progress_remaining(
                    policy.num_timesteps, total_timesteps
                )
                
                # 记录metrics
                if log_interval is not None and num_timesteps % log_interval == 0:
                    fps = int(policy.num_timesteps / (time.time() - policy.start_time))
                    wandb.log({f"{polid}/fps": fps}, step=num_timesteps)
                    ep_rew_mean = safe_mean([ep_info["r"] for ep_info in policy.ep_info_buffer])
                    wandb.log({f"{polid}/ep_rew_mean": ep_rew_mean}, step=num_timesteps)
                    SW_ep_rew_mean += ep_rew_mean
                    wandb.log({f"{polid}/ep_len_mean": policy.ep_info_buffer[-1]["l"]}, step=num_timesteps)
                    wandb.log({f"{polid}/time_elapsed": int(time.time() - policy.start_time)}, step=num_timesteps)
                    wandb.log({f"{polid}/total_timesteps": policy.num_timesteps}, step=num_timesteps)

                    if hasattr(policy.rollout_buffer, 'cf_rewards'):
                        ep_cf_reward = np.sum(policy.rollout_buffer.cf_rewards)
                        wandb.log({f"{polid}/cf_reward": ep_cf_reward}, step=num_timesteps)
                    
                    policy.logger.record("policy_id", polid, exclude="tensorboard")
                    policy.logger.record("time/iterations", num_timesteps, exclude="tensorboard")
                    
                    if len(policy.ep_info_buffer) > 0 and len(policy.ep_info_buffer[0]) > 0:
                        policy.logger.record(
                            "rollout/ep_rew_mean",
                            safe_mean([ep_info["r"] for ep_info in policy.ep_info_buffer])
                        )
                        policy.logger.record(
                            "rollout/ep_len_mean",
                            safe_mean([ep_info["l"] for ep_info in policy.ep_info_buffer])
                        )
                    
                    policy.logger.record("time/fps", fps)
                    policy.logger.record(
                        "time/time_elapsed",
                        int(time.time() - policy.start_time),
                        exclude="tensorboard"
                    )
                    policy.logger.record(
                        "time/total_timesteps",
                        policy.num_timesteps,
                        exclude="tensorboard"
                    )
                    policy.logger.dump(step=policy.num_timesteps)

                # 训练
                policy.train()
            
            wandb.log({"SW_ep_rew_mean": SW_ep_rew_mean/self.num_agents}, step=num_timesteps)
            wandb.log({"SW_ep_rew_total": SW_ep_rew_mean}, step=num_timesteps)

        for callback in callbacks:
            callback.on_training_end()

    def collect_rollouts(self, last_obs, callbacks, num_timesteps):
        all_last_episode_starts = [None] * self.num_agents
        all_obs = [None] * self.num_agents
        all_last_obs = [None] * self.num_agents
        all_rewards = [None] * self.num_agents
        all_dones = [None] * self.num_agents
        all_infos = [None] * self.num_agents
        all_distributions = [None] * self.num_agents

        all_obs_trajs = []
        all_actions_trajs = []
        all_rewards_trajs = []
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
                    actions, values, log_probs, distributions = policy.policy.forward(obs_tensor)
                    
                    all_actions[polid] = actions
                    all_values[polid] = values
                    all_log_probs[polid] = log_probs
                    all_distributions[polid] = distributions
                    
                    clipped_actions = actions.cpu().numpy()
                    if isinstance(self.action_space, Box):
                        clipped_actions = np.clip(
                            clipped_actions,
                            self.action_space.low,
                            self.action_space.high,
                        )
                    elif isinstance(self.action_space, Discrete):
                        clipped_actions = np.array([action.item() for action in clipped_actions])
                    all_clipped_actions[polid] = clipped_actions
                
            all_clipped_actions = np.vstack(all_clipped_actions).transpose().reshape(-1)
            
            obs, rewards, dones, infos = self.env.step(all_clipped_actions)
            
            all_obs_trajs.append([obs.copy() for obs in all_last_obs])
            all_actions_trajs.append([action.cpu().numpy() for action in all_actions])
            all_rewards_trajs.append([reward for reward in all_rewards if reward is not None])
            
            for polid in range(self.num_agents):
                all_obs[polid] = np.array(
                    [obs[envid * self.num_agents + polid] for envid in range(self.num_envs)]
                )
                all_rewards[polid] = np.array(
                    [rewards[envid * self.num_agents + polid] for envid in range(self.num_envs)]
                )
                all_dones[polid] = np.array(
                    [dones[envid * self.num_agents + polid] for envid in range(self.num_envs)]
                )
                all_infos[polid] = [infos[envid * self.num_agents + polid] for envid in range(self.num_envs)]

            for policy in self.policies:
                policy.num_timesteps += self.num_envs

            for callback in callbacks:
                callback.update_locals(locals())
                if not callback.on_step():
                    return obs

            for polid, policy in enumerate(self.policies):
                policy._update_info_buffer(all_infos[polid])

            steps += 1

            for polid, policy in enumerate(self.policies):
                if isinstance(self.action_space, Discrete):
                    all_actions[polid] = all_actions[polid].reshape(-1, 1)
                all_actions[polid] = all_actions[polid].cpu().numpy()

            for polid, policy in enumerate(self.policies):
                if self.model == 'social_influence':
                    if len(all_obs_trajs) > 10:
                        social_influence = self.compute_social_influence(
                            policy,
                            all_obs_trajs[-10:],
                            all_last_obs,
                            all_actions,
                            polid
                        )
                    else:
                        social_influence = np.zeros((self.num_envs,1))
                    
                    policy.rollout_buffer.add_sw(
                        all_last_obs[polid],
                        all_actions[polid],
                        all_rewards[polid],
                        all_last_episode_starts[polid],
                        all_values[polid],
                        all_log_probs[polid],
                        all_last_obs,
                        all_actions,
                        all_rewards,
                        social_influence,
                    )
            
            all_last_obs = all_obs
            all_last_episode_starts = all_dones

        with th.no_grad():
            for polid, policy in enumerate(self.policies):
                obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                _, value, _, _ = policy.policy.forward(obs_tensor)
                if self.model == 'social_influence':
                    policy.rollout_buffer.compute_social_influence_returns_and_advantage(
                        last_values=value,
                        dones=all_dones[polid],
                        alpha=self.alpha
                    )

        for callback in callbacks:
            callback.on_rollout_end()

        for polid, policy in enumerate(self.policies):
            policy._last_episode_starts = all_last_episode_starts[polid]

        return obs

    def compute_social_influence(self, policy, all_obs_trajs, all_last_obs, all_actions, polid):
        """计算社会影响力
        
        Args:
            policy: 当前智能体的策略
            all_obs_trajs: 所有智能体的观察轨迹历史 [T, num_agents, ...]
            all_last_obs: 所有智能体的最新观察 [num_agents, ...]
            all_actions: 所有智能体的动作 [num_agents, ...]
            polid: 当前智能体的ID
            
        Returns:
            social_influence: 社会影响力得分 [num_envs, 1]
        """
        with th.no_grad():
            # 将数据转移到GPU
            all_last_obs = th.from_numpy(np.array(all_last_obs)).to(policy.device)
            all_actions = th.from_numpy(np.transpose(np.array(all_actions),(1,0,2))).to(policy.device)
            
            # 处理观察轨迹
            batch_obs_tensor = th.from_numpy(np.array(all_obs_trajs)).to(policy.device)
            # [T, num_agents, num_envs, ...] -> [num_agents, num_envs, T, ...]
            batch_obs_tensor = batch_obs_tensor.permute(1, 2, 0, 3, 4, 5)
            
            # 提取特征
            obs_features_list = []
            for agent_obs in batch_obs_tensor:
                # 重塑以批量处理时序数据
                agent_obs = agent_obs.reshape(-1, *agent_obs.shape[2:])
                feat = policy.policy.extract_features(agent_obs)
                # 恢复时序维度 [num_envs, T, feature_dim]
                feat = feat.reshape(batch_obs_tensor.shape[1], -1, feat.shape[-1])
                obs_features_list.append(feat)
            
            # [num_agents, num_envs, T, feature_dim]
            obs_features = th.stack(obs_features_list)
            
            total_kl_div = th.zeros(self.num_envs, device=policy.device)
            
            # 对每个其他智能体计算KL散度
            for j in range(self.num_agents):
                if j != polid:
                    # 使用LSTM提取时序特征
                    outputs = policy.policy.lstm_extractor(
                        obs_features[j],  # [num_envs, T, feature_dim]
                        self.prev_state
                    )
                    p_a_given_s, _, p_a_given_m_s, _, self.prev_state = outputs
                    
                    # 计算KL散度
                    kl_div = F.kl_div(
                        p_a_given_m_s.log(),  # 预测的动作分布
                        p_a_given_s,          # 实际的动作分布
                        reduction='none'       # 不要立即求平均
                    )
                    
                    # 对每个环境分别累加KL散度
                    total_kl_div += kl_div.mean(dim=1)  # 对动作维度求平均
            
            # 计算平均社会影响力
            avg_kl_div = total_kl_div / (self.num_agents - 1)  # 除去自身
            
            # 返回每个环境的社会影响力得分
            social_influence = avg_kl_div.reshape(-1, 1).cpu().numpy()
            
            # 应用sigmoid来归一化得分到[0,1]区间
            social_influence = 1 / (1 + np.exp(-social_influence))
            
            return social_influence

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
    ) -> "Causal_IndependentPPO":
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