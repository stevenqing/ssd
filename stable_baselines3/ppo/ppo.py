import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces #from gymnasium import spaces
from torch.nn import functional as F
import wandb
# from social_dilemmas.ppo.AgA import AgA_Grad
from torch.optim import Adam
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from collections import defaultdict
from torch.nn.utils import parameters_to_vector
SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        eval_env: Optional[GymEnv],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        aga = True,
        _init_setup_model: bool = True,
        eval_interval: int = 4000,
        eval_episodes: int = 1,
        env_nums=0, 
        agent_nums=0,
        l=1, 
        lr_schedule_args=None
    ):  
        super().__init__(
            policy,
            env,
            eval_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            aga=aga,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
            eval_interval = eval_interval,
            eval_episodes = eval_episodes,
        )
        self.lr_schedule_args = lr_schedule_args
        self.l=l
        self.env_nums = env_nums
        self.agent_nums = agent_nums
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self.optimizer = th.optim.Adam(self.policy.parameters())
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
            
    def reshape_flatten(self, data):
        shape = data.shape
        return data.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
    
    def scheduler(self, l_schedule_args, steps):
        l = l_schedule_args["l"]
        alpha = steps // l_schedule_args["steps"]
        return l * l_schedule_args["weight"] ** alpha
    
    def train(self, policy_gradient=None, aga=False) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # if aga==True:
        # Switch to train mode (this affects batch norm / dropout)
        # Update optimizer learning rate
        self._update_learning_rate(self.optimizer, self.num_timesteps, self.lr_schedule_args)
        # self._update_learning_rate(self.policy.actor_optimizer, self.num_timesteps, self.lr_schedule_args)
        # self._update_learning_rate(self.policy.critic_optimizer, self.num_timesteps, self.lr_schedule_args)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        value_losses = []
        policy_losses = []
        clip_fractions = []
        # pg_losses = defaultdict(list)
        pg_losses = []
        SW_pg_losses = []
        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            # policy_loss, SW_policy_loss, other_loss, entropy_loss = self.cal_policy_loss()
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Author: Yang Li
                # Time: 2024/01/06
                # Description: We need first to flatten data, which should be operated in buffer sample function
                # observations: th.Tensor
                # actions: th.Tensor
                # old_values: th.Tensor
                # old_log_prob: th.Tensor
                # advantages: th.Tensor
                # returns: th.Tensor
                # SW_returns: th.Tensor
                # SW_values: th.Tensor
                # shaped_rewards: th.Tensor
                #  arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
                
                # the original shape is torch.Size([100 (batch size), 60 ([agent 1, ..., 5 on env1, 1,2... on env2,....]), *])
                # convert to envs * agents * times
                # Shuqing: original shape [batch_size, num_agents, view_window(15), view_window(15), channel]
                observations = self.reshape_flatten(rollout_data.observations)
                old_values = self.reshape_flatten(rollout_data.old_values).flatten()
                old_log_prob = self.reshape_flatten(rollout_data.old_log_prob).flatten()
                advantages = self.reshape_flatten(rollout_data.advantages).flatten()
                shaped_advantages = self.reshape_flatten(rollout_data.shaped_advantages).flatten()
                returns = self.reshape_flatten(rollout_data.returns).flatten()
                # shaped_rewards = self.reshape_flatten(rollout_data.shaped_rewards).flatten()
                actions = self.reshape_flatten(rollout_data.actions) 
                
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()
                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values,  log_prob, entropy = self.policy.evaluate_actions(observations, actions)
                values = values.flatten()
                # Normalize advantage
                # print(rollout_data)
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                if self.normalize_advantage and len(shaped_advantages) > 1:
                    shaped_advantages = (shaped_advantages - shaped_advantages.mean()) / (shaped_advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - old_log_prob)

                # clipped surrogate loss
                # policy_loss_1 = advantages * ratio
                # policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                # policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # # Logging
                # pg_losses.append(policy_loss.item())
                # clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                # clip_fractions.append(clip_fraction)

                # if self.clip_range_vf is None:
                #     # No clipping
                #     values_pred = values
                # else:
                #     # Clip the difference between old and new value
                #     # NOTE: this depends on the reward scaling
                #     values_pred = rollout_data.old_values + th.clamp(
                #         values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                #     )
                # # Value loss using the TD(gae_lambda) target
                # value_loss = F.mse_loss(th.flatten(rollout_data.returns), values_pred)
                # value_losses.append(value_loss.item())

                # # Entropy loss favor exploration
                # if entropy is None:
                #     # Approximate entropy when no analytical form
                #     entropy_loss = -th.mean(-log_prob)
                # else:
                #     entropy_loss = -th.mean(entropy)

                # entropy_losses.append(entropy_loss.item())

                # loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # with th.no_grad():
                #     log_ratio = log_prob - th.flatten(rollout_data.old_log_prob)
                #     approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                #     approx_kl_divs.append(approx_kl_div)

                # if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                #     continue_training = False
                #     if self.verbose >= 1:
                #         print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                #     break

                # # Optimization step
                # self.optimizer.zero_grad()
                # loss.backward()
                # # Clip grad norm
                # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # self.optimizer.step()
            
                '''
                sw loss calculation
                '''
                # # clipped surrogate loss
                # # Author: Yang Li
                # # Time: 2024/01/09
                # # Description: Yang Li Calculate SW advantages using total objectives
                # SW_policy_loss_1 = shaped_advantages * ratio
                # SW_policy_loss_2 = shaped_advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            
                # SW_policy_loss = -th.min(SW_policy_loss_1, SW_policy_loss_2).mean()

                # # Author: Yang Li
                # # Time: 2024/01/09
                # # Description: Yang Li Calculate SW advantages using total objectives
                # policy_loss_1 = advantages * ratio
                # policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            
                # policy_loss_1_agents = [None] * self.agent_nums
                # for poid in range(self.agent_nums):
                #     tmp = [None] * self.env_nums
                #     for env_id in range(self.env_nums):
                #         tmp[env_id] = policy_loss_1[self.agent_nums * self.batch_size * env_id + poid * self.batch_size : self.agent_nums * self.batch_size * env_id + (poid+1)*self.batch_size]
                #     policy_loss_1_agents[poid] = th.stack(tmp)
    
                # policy_loss_2_agents = [None] * self.agent_nums
                # for poid in range(self.agent_nums):
                #     tmp = [None] * self.env_nums
                #     for env_id in range(self.env_nums):
                #         tmp[env_id] = policy_loss_2[self.agent_nums * self.batch_size * env_id + poid * self.batch_size : self.agent_nums * self.batch_size * env_id + (poid+1)*self.batch_size]
                #     policy_loss_2_agents[poid] = th.stack(tmp)
                
                # # policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                # policy_loss_agents = [None] * self.agent_nums
                # for poid in range(self.agent_nums):
                #     policy_loss_agents[poid] = -th.min(policy_loss_1_agents[poid], policy_loss_2_agents[poid]).mean()
                
                # # Author: Yang Li
                # # Time: 2024/01/09
                # # Description: For homogeneous, we use mean of polic loss agents.
                # policy_loss = th.stack(policy_loss_agents).mean()
                
                # if entropy is None:
                #     # Approximate entropy when no analytical form
                #     entropy_loss = -th.mean(-log_prob)
                # else:
                #     entropy_loss = -th.mean(entropy)
                
                # # Logging
                # clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                # clip_fractions.append(clip_fraction)

                # if self.clip_range_vf is None:
                #     # No clipping
                #     values_pred = values
                # else:
                #     # Clip the difference between old and new value
                #     # NOTE: this depends on the reward scaling
                #     values_pred = old_values + th.clamp(
                #         values - old_values, -clip_range_vf, clip_range_vf
                #     )
                # # Value loss using the TD(gae_lambda) target
                # value_loss = F.mse_loss(returns, values_pred)
                
                # # print(SW_policy_loss.item())
                # value_losses.append(value_loss.item())
                # for poid in range(len(policy_loss_agents)):
                #     pg_losses[poid].append(policy_loss_agents[poid].item())
                # #     print(f"\t {policy_loss_agents[poid].item()}")
                # # print(f"Mean \t {th.stack(policy_loss_agents).mean()}")
                # SW_pg_losses.append(SW_policy_loss.item())
                # entropy_losses.append(entropy_loss.item())
                # policy_losses.append(policy_loss.item())

                # # Calculate approximate form of reverse KL Divergence for early stopping
                # # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # # and Schulman blog: http://joschu.net/blog/kl-approx.html
                # with th.no_grad():
                #     log_ratio = log_prob - old_log_prob
                #     approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                #     approx_kl_divs.append(approx_kl_div)

                # if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                #     continue_training = False
                #     if self.verbose >= 1:
                #         print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                #     break

                # # Optimization step
                # #update critic loss
                # self.policy.critic_optimizer.zero_grad()
                # (self.vf_coef * value_loss).backward(retain_graph=True)
                # # Clip grad norm
                # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # self.policy.critic_optimizer.step()
                
                # # Yang Li use new 
                # if aga:
                #     self.policy.actor_optimizer.zero_grad()
                #     l = self.scheduler(self.l_schedule_args, self.num_timesteps)
                #     # Author: Yang Li
                #     # Time: 2024/01/21
                #     # Description: ATTENTION this version is ablation study without sign alignment!
                #     policy_gradient = AgA_Grad(policy_loss, self.ent_coef * entropy_loss + SW_policy_loss, self.policy, l, False)
                #     # for poid in range(0, self.agent_nums):
                #     self.policy.actor_optimizer.zero_grad()
                #         # for i, p in enumerate(self.policy.action_net.parameters()):
                #         #     print(p.grad)
                #         #     break
                #     for i, p in enumerate(self.policy.action_net.parameters()):
                #         p.grad = policy_gradient[i]
                #         # for i, p in enumerate(self.policy.action_net.parameters()):
                #         #     print(p.grad)
                #         #     break
                #         # exit()
                #     th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                #     self.policy.actor_optimizer.step()
                # else:
                #     # else use consensus optimization
                #     self.policy.actor_optimizer.zero_grad()
                #     xi_jt_grad = th.autograd.grad(SW_policy_loss,
                #                         self.policy.parameters(),
                #                         create_graph=False,
                #                         allow_unused=True) 
                #     xi_jt_grad_vec = parameters_to_vector(xi_jt_grad)
                #     # calculate H=||xi||^2/2
                #     gH = th.norm(xi_jt_grad_vec)**2/2
            
                #     self.policy.actor_optimizer.zero_grad()
                #     (self.ent_coef * entropy_loss + SW_policy_loss + gH).backward(allow_unused=True)
                #     self.policy.actor_optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        for poid in range(len(pg_losses)):
            wandb.log({f"train/policy_gradient_loss/{poid}": np.mean(pg_losses[poid])}, step=self.num_timesteps)
            # wandb.log({f"train/other_loss/{poid}": np.mean(other_losses[poid])}, step=self.num_timesteps)
        
        wandb.log({f"train/entropy_loss{poid}": np.mean(entropy_losses)}, step=self.num_timesteps)
        # wandb.log({f"train/l": l}, step=self.num_timesteps)
        wandb.log({f"train/value_loss": np.mean(value_losses)}, step=self.num_timesteps)
        wandb.log({f"train/approx_kl": np.mean(approx_kl_divs)}, step=self.num_timesteps)
        wandb.log({f"train/clip_fraction": np.mean(clip_fraction)}, step=self.num_timesteps)
        wandb.log({f"train/loss": loss}, step=self.num_timesteps)
        wandb.log({f"train/explained_variance": explained_var}, step=self.num_timesteps)
        if hasattr(self.policy, "log_std"):
            wandb.log({f"train/std": th.exp(self.policy.log_std).mean().item()}, step=self.num_timesteps)

        # self.logger.record("train/loss", np.mean(SW_pg_losses), exclude="tensorboard")
        
        wandb.log({f"train/clip_range": clip_range}, step=self.num_timesteps)
        if self.clip_range_vf is not None:
            #self.logger.record("train/clip_range_vf", clip_range_vf)
            wandb.log({f"train/clip_range_vf": clip_range_vf}, step=self.num_timesteps)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        num_envs: int = 0,
        num_agents: int = 0,
        l_schedule_args=None
    ) -> SelfPPO:
        self.l_schedule_args = l_schedule_args
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            num_envs = num_envs,
            num_agents = num_agents
        )