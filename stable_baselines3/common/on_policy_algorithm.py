import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
import wandb
from gym import spaces #from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        aga: bool = False,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        eval_interval = 4000,
        eval_episodes = 1
    ):
        super().__init__(
            policy=policy,
            env=env,
            eval_env=eval_env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        self.aga = aga
        # self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        num_envs: int,
        num_agents: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            # print(self.policy)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            # print(clipped_actions, clipped_actions.shape)
            # print(env.step)
            # 1/0

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            all_rewards = []
            for agentid in range(num_agents):
                all_rewards.append(np.array([rewards[envid * num_agents + agentid] for envid in range(num_envs)]))
            # for envid in range(num_envs):
            #     all_rewards[envid].append(list(rewards[envid * num_agents : (envid + 1) * num_agents]))
            all_rewards = np.array(all_rewards)
            svo = True
            if svo:
                all_rewards = []
                for agentid in range(num_agents):
                    all_rewards.append(np.array([rewards[envid * num_agents + agentid] for envid in range(num_envs)]))
                # for envid in range(num_envs):
                #     all_rewards[envid].append(list(rewards[envid * num_agents : (envid + 1) * num_agents]))
                all_rewards = np.array(all_rewards)
                sum_r = sum(all_rewards)
                tanh = [None] * num_agents
                shape_rewards = []
                for polid in range(num_agents):
                    tanh[polid] = ((sum_r - all_rewards[polid])/(num_agents-1)) / (1e-10+all_rewards[polid])
                theta = np.arctan(tanh)
                target_theta = np.ones_like(theta)
                SVO_value = np.abs(target_theta - theta)
                for envid in range(num_envs):
                    for polid in range(num_agents):
                        shape_rewards.extend([SVO_value[polid, envid]])
                shape_rewards = rewards - 0.01 * np.array(shape_rewards)
            shape_rewards = np.array(shape_rewards)
            # the rewards.. are [agent0, agent1, agent2, agent4, agent0, agent1,....]
            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                shape_rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def cal_equality_metric(self, rewards):
        num_agents = len(rewards)
        mean_r = np.mean(rewards)
        rank_r = sorted(rewards)
        value = 0
        for i in range(num_agents):
            value += (i+1)*(rank_r[i]-mean_r)
        value = 2* value / (num_agents**2*mean_r)
        
        return value

# def cal_eq(arrs):
#     num_agents = arrs.shape[0]
#     times = arrs.shape[1]
#     values = []
#     for i in range(times):
#         arr = arrs[:, i]
#         mean_arr = np.mean(arr)
#         rank_arr = sorted(arr)
#         value = 0
#         for i in range(num_agents):
#             value += (i+1)*(rank_arr[i]-mean_arr)
#         value = 2* value / (num_agents**2*mean_arr)
#         if value < 0:
#             value = 0
#         # value = value / (abs(mean_arr))
#         values.append(value)
#     return values
    
    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        num_envs: int = 0,
        num_agents: int = 0,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, num_envs=num_envs, num_agents=num_agents)

            if not continue_training:
                break
            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                rew_by_policy = [0 for _ in range(num_agents)]
                for i in range(num_agents):
                    rew_by_policy[i] = safe_mean([self.ep_info_buffer[env_id * num_agents + i]["r"] for env_id in range(num_envs)])

                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                #self.logger.record("time/iterations", iteration, exclude="tensorboard")
                wandb.log({f"time/iterations": iteration}, step=self.num_timesteps)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    if iteration % 1000 == 0:
                        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                        self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                        self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                        for i in range(num_agents):
                            self.logger.record(f"rollout/ep_rew_agent{i}", rew_by_policy[i])
                    wandb.log({f"rollout/ep_rew/SW_mean": num_agents * safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])}, step=self.num_timesteps)
                    for i in range(num_agents):
                        wandb.log({f"rollout/ep_rew/mean_agent{i}": rew_by_policy[i]}, step=self.num_timesteps)
                    
                    wandb.log({f"rollout/ep_len_mean": safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])}, step=self.num_timesteps)
                
                wandb.log({f"rollout/equality": self.cal_equality_metric(rew_by_policy)}, step=self.num_timesteps)
                #self.logger.record ("time/fps", fps)
                wandb.log({f"time/fps": fps}, step=self.num_timesteps)
                #self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                wandb.log({f"time/time_elapsed": int(time_elapsed)}, step=self.num_timesteps)
                wandb.log({f"time/total_timesteps": self.num_timesteps}, step=self.num_timesteps)
                
                self.logger.dump(step=self.num_timesteps)

            self.train()

            # if self.num_timesteps % self.eval_interval == 0:
            #     self.eval(self.env, callback, iteration, num_envs, num_agents)

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        # if not self.aga:
        state_dicts = ["policy", "policy.optimizer"]
        # else:
            # state_dicts = ["policy", "policy.actor_optimizer", "policy.critic_optimizer"]

        return state_dicts, []
    
    def eval(self, env, callback, iteration, num_envs, num_agents):
        self.collect_rollouts(env, callback, self.rollout_buffer, n_rollout_steps=self.eval_episodes, num_envs=num_envs, num_agents=num_agents)
        wandb.log({f"eval/ep_rew_mean": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])}, step=iteration)
        # print()

