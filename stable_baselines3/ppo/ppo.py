import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union
import random
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F
import wandb
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy, RewardActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.utils import (configure_logger, obs_as_tensor,
                                            safe_mean)
SelfPPO = TypeVar("SelfPPO", bound="PPO")

from social_dilemmas.envs.agent import ENV_REWARD_SPACE, OOD_INDEX
import copy

REWARD_ENV_SPACE = {"harvest": {value: key for key, value in ENV_REWARD_SPACE["harvest"].items()}}

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
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
        "RewardPolicy": RewardActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
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
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        model: str = 'baseline',
        num_agents: int = 3,
        enable_trajs_learning: bool = False,
        polid: Optional[int] = None,
        env_name: Optional[str] = 'harvest'
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

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
        self.model = model
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.num_agents = num_agents
        self.enable_trajs_learning = enable_trajs_learning
        self.polid = polid
        self.env_name = env_name
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        self.policy.num_agents = self.num_agents



    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, reward_losses, reweighted_reward_losses = [], [], [], None
        clip_fractions = []
        
        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer

            if self.model == 'causal':
                if self.enable_trajs_learning:
                    for rollout_data in self.rollout_buffer.get_sw_traj(self.batch_size): # calculate the loss for each batcn
                        all_last_obs = rollout_data.all_last_obs
                        all_rewards = rollout_data.all_rewards
                        actions = rollout_data.actions
                        if isinstance(self.action_space, spaces.Discrete):
                            # Convert discrete action from float to long
                            actions = rollout_data.actions.long().flatten()
                            all_actions = rollout_data.all_actions.long()
                        # Re-sample the noise matrix because the log_std has changed
                        if self.use_sde:
                            self.policy.reset_noise(self.batch_size)
                        

                        values, log_prob, entropy, predicted_reward = self.policy.evaluate_actions(rollout_data.observations, actions, all_last_obs, all_actions)
                        
                        # enabling traj learning
                        reweighted_reward_losses = 0
                        if not th.equal(rollout_data.prev_action_traj,rollout_data.all_action_traj):
                            prev_obs_traj = rollout_data.prev_obs_traj
                            prev_actions_traj = rollout_data.prev_action_traj
                            prev_rewards_traj = rollout_data.prev_rewards_traj
                            all_obs_traj = rollout_data.all_obs_traj
                            all_actions_traj = rollout_data.all_action_traj
                            all_rewards_traj = rollout_data.all_rewards_traj
                            
                            reweighted_actions_list,reweighted_action_index,equal_weight_list,reweighted_shape = [],None,[],[]
                            for i in range(self.action_space.n):
                                tmp = (prev_actions_traj == i).sum() - (all_actions_traj == i).sum()
                                reweighted_actions_list.append(int(tmp))
                                # (action_sapce, difference between prev and now action)
                                if reweighted_action_index == None:
                                    reweighted_action_index = th.nonzero(all_actions_traj == i)
                                else:
                                    reweighted_action_index = th.cat([reweighted_action_index,th.nonzero(all_actions_traj == i)],0)
                                reweighted_shape.append(th.nonzero(all_actions_traj == i).shape[0])

                                shape = th.nonzero(all_actions_traj == i).shape[0]
                                equal_weight = [1/shape for _ in np.arange(shape)]
                                # (action_space, fot each difference list random sampling)
                                equal_weight_list.append(equal_weight)
                            if int(prev_rewards_traj.sum()) < int(all_rewards_traj.sum()):
                                reweighted_actions_list = [-x for x in reweighted_actions_list]
                                reweighted_actions_list = [x if x >= 0 else 0 for x in reweighted_actions_list]
                            else:
                                reweighted_actions_list = [x if x >= 0 else 0 for x in reweighted_actions_list]
                            reweighted_actions_prob = th.tensor([reweighted_actions_list[x]/(sum(reweighted_actions_list) * reweighted_shape[x]) for x in range(len(reweighted_actions_list))]) # within 8
                            total_reweighted_actions_prob = None
                            for i in range(self.action_space.n):
                                tmp = reweighted_actions_prob[i].repeat(reweighted_shape[i])
                                if total_reweighted_actions_prob == None:
                                    total_reweighted_actions_prob = tmp
                                else:
                                    total_reweighted_actions_prob = th.cat([total_reweighted_actions_prob,tmp])
                            sample_index = th.utils.data.WeightedRandomSampler(total_reweighted_actions_prob,self.batch_size,replacement=True)
                            total_action_index = reweighted_action_index[list(sample_index)]
                            reweighted_index = total_action_index[:,0]

                            reweighted_obs = all_obs_traj[reweighted_index]
                            reweighted_actions = all_actions_traj[reweighted_index].unsqueeze(-1)

                            reweighted_add_reward = (int(prev_rewards_traj.sum()) - int(all_rewards_traj.sum()))/self.batch_size
                            reweighted_reward = all_rewards_traj.cuda().view(-1,1).scatter(0,th.tensor(list(sample_index)).cuda().unsqueeze(-1),reweighted_add_reward,reduce='add')
                            reweighted_reward = reweighted_reward.view(self.batch_size,-1)

                            predicted_trajs_reweighted_reward = self.policy.get_trajs_reweighted_reward(reweighted_obs,reweighted_actions)
                            reweighted_reward_losses = F.mse_loss(reweighted_reward, predicted_trajs_reweighted_reward)

                            # actions_list = np.arange(self.action_space.n)
                            # batch_reweighted_actions = np.random.choice(actions_list, self.batch_size, p=reweighted_actions_prob)
                                                       
                            # reweighted_index = [np.random.choice(list(np.array(reweighted_action_index[x][:,0].cpu())),1,equal_weight_list[x]) for x in batch_reweighted_actions]
                            # reweighted_obs = th.permute(all_obs_traj[np.array(reweighted_index)],(0,2,1,3,4,5)).squeeze()
                            # reweighted_actions = th.permute(all_actions_traj[np.array(reweighted_index)],(0,2,1))
                            # reweighted_rewards = th.permute(all_rewards_traj[np.array(reweighted_index)],(0,2,1)).squeeze()
                            
                            # add_reward_agent_index = [random.choice(th.where(reweighted_actions[x] == batch_reweighted_actions[x])[0].tolist()) for x in batch_reweighted_actions]
                            # reweighted_add_reward = (int(prev_rewards_traj.sum()) - int(all_rewards_traj.sum()))/self.batch_size
                            # for x in range(self.batch_size):
                            #     reweighted_rewards[x][add_reward_agent_index[x]] += reweighted_add_reward

                        
                        wandb.log({f"train/predicted_reward": predicted_reward}, step=self.num_timesteps)
                        values = values.flatten()
                        # Normalize advantage
                        advantages = rollout_data.advantages
                        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                        if self.normalize_advantage and len(advantages) > 1:
                            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                        # ratio between old and new policy, should be one at the first iteration
                        ratio = th.exp(log_prob - rollout_data.old_log_prob)

                        # clipped surrogate loss
                        policy_loss_1 = advantages * ratio
                        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                        # Logging
                        pg_losses.append(policy_loss.item())
                        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                        clip_fractions.append(clip_fraction)

                        if self.clip_range_vf is None:
                            # No clipping
                            values_pred = values
                        else:
                            # Clip the difference between old and new value
                            # NOTE: this depends on the reward scaling
                            values_pred = rollout_data.old_values + th.clamp(
                                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                            )
                        # Value loss using the TD(gae_lambda) target
                        value_loss = F.mse_loss(rollout_data.returns, values_pred)
                        value_losses.append(value_loss.item())

                        # Entropy loss favor exploration
                        if entropy is None:
                            # Approximate entropy when no analytical form
                            entropy_loss = -th.mean(-log_prob)
                        else:
                            entropy_loss = -th.mean(entropy)

                        entropy_losses.append(entropy_loss.item())

                        # Reward loss
                        reward_losses = F.mse_loss(all_rewards, predicted_reward)
                        if reweighted_reward_losses != 0:
                            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + reward_losses + reweighted_reward_losses
                        else:
                            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + reward_losses

                        # Calculate approximate form of reverse KL Divergence for early stopping
                        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                        # and Schulman blog: http://joschu.net/blog/kl-approx.html
                        with th.no_grad():
                            log_ratio = log_prob - rollout_data.old_log_prob
                            approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                            approx_kl_divs.append(approx_kl_div)

                        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                            continue_training = False
                            if self.verbose >= 1:
                                print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                            break

                        # Optimization step
                        self.policy.optimizer.zero_grad()
                        loss.backward()
                        # Clip grad norm
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.policy.optimizer.step()
                else:
                    for rollout_data in self.rollout_buffer.get_sw(self.batch_size):
                        all_last_obs = rollout_data.all_last_obs
                        all_rewards = rollout_data.all_rewards
                        actions = rollout_data.actions
                        if isinstance(self.action_space, spaces.Discrete):
                            # Convert discrete action from float to long
                            actions = rollout_data.actions.long().flatten()
                            all_actions = rollout_data.all_actions.long()
                        # Re-sample the noise matrix because the log_std has changed
                        if self.use_sde:
                            self.policy.reset_noise(self.batch_size)

                        values, log_prob, entropy, predicted_reward = self.policy.evaluate_actions(rollout_data.observations, actions, all_last_obs, all_actions)
                        values = values.flatten()
                        # Normalize advantage
                        advantages = rollout_data.advantages
                        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                        if self.normalize_advantage and len(advantages) > 1:
                            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                        # ratio between old and new policy, should be one at the first iteration
                        ratio = th.exp(log_prob - rollout_data.old_log_prob)

                        # clipped surrogate loss
                        policy_loss_1 = advantages * ratio
                        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                        # Logging
                        pg_losses.append(policy_loss.item())
                        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                        clip_fractions.append(clip_fraction)

                        if self.clip_range_vf is None:
                            # No clipping
                            values_pred = values
                        else:
                            # Clip the difference between old and new value
                            # NOTE: this depends on the reward scaling
                            values_pred = rollout_data.old_values + th.clamp(
                                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                            )
                        # Value loss using the TD(gae_lambda) target
                        value_loss = F.mse_loss(rollout_data.returns, values_pred)
                        value_losses.append(value_loss.item())

                        # Entropy loss favor exploration
                        if entropy is None:
                            # Approximate entropy when no analytical form
                            entropy_loss = -th.mean(-log_prob)
                        else:
                            entropy_loss = -th.mean(entropy)

                        entropy_losses.append(entropy_loss.item())



                        # use discrete loss
                        all_rewards = all_rewards.long()
                        predicted_reward = th.permute(predicted_reward,(0,2,1))
                        # TODO use logits here, make sure flatten first
                        # TODO check the data flatten in a right way
                        # predicted_reward =  predicted_reward.reshape(-1, predicted_reward.shape[-1])
                        # all_rewards = all_rewards.reshape(-1)
                        reward_losses = F.cross_entropy(predicted_reward, all_rewards)
                        reward_acc = th.mean((th.argmax(predicted_reward, dim=1) == all_rewards).float()).cpu().numpy()
                        
                        # acc 
                        reward_ood_rate = th.mean((all_rewards == OOD_INDEX[self.env_name][0]).float()).cpu().numpy()
                        # reward_losses = F.mse_loss(all_rewards, predicted_reward)

                        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + reward_losses

                        # Calculate approximate form of reverse KL Divergence for early stopping
                        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                        # and Schulman blog: http://joschu.net/blog/kl-approx.html
                        with th.no_grad():
                            log_ratio = log_prob - rollout_data.old_log_prob
                            approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                            approx_kl_divs.append(approx_kl_div)

                        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                            continue_training = False
                            if self.verbose >= 1:
                                print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                            break

                        # Optimization step
                        self.policy.optimizer.zero_grad()
                        loss.backward()
                        # Clip grad norm
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.policy.optimizer.step()
            elif self.model == 'team':
                for rollout_data in self.rollout_buffer.get_sw(self.batch_size):
                    all_last_obs = rollout_data.all_last_obs
                    all_rewards = rollout_data.all_rewards
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()
                        all_actions = rollout_data.all_actions.long()
                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    values, log_prob, entropy, predicted_reward = self.policy.evaluate_actions(rollout_data.observations, actions, all_last_obs, all_actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())



                    # use discrete loss
                    # reward_losses = F.mse_loss(all_rewards, predicted_reward)

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
            else:
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()


            if not continue_training:
                break

                

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        
        wandb.log({f"train/entropy_loss": np.mean(entropy_losses)}, step=self.num_timesteps)
        wandb.log({f"train/value_loss": np.mean(value_losses)}, step=self.num_timesteps)
        wandb.log({f"train/approx_kl": np.mean(approx_kl_divs)}, step=self.num_timesteps)
        wandb.log({f"train/clip_fraction": np.mean(clip_fraction)}, step=self.num_timesteps)
        wandb.log({f"train/explained_variance": explained_var}, step=self.num_timesteps)
        if hasattr(self.policy, "log_std"):
            wandb.log({f"train/std": th.exp(self.policy.log_std).mean().item()}, step=self.num_timesteps)

        if self.model == 'causal':
            wandb.log({f"train/reward_loss": reward_losses.item()}, step=self.num_timesteps)
            if self.enable_trajs_learning == False:
                if self.polid != None:
                    output_predicted_reward = th.argmax(predicted_reward, dim=1)
                    reverse_reward_mapping_func = np.frompyfunc(lambda key: REWARD_ENV_SPACE[self.env_name].get(key, OOD_INDEX[self.env_name][1]), 1, 1) #SPEED, Can the function be jit?
                    all_rewards_values = np.mean(reverse_reward_mapping_func(all_rewards.cpu().numpy()),axis=0)
                    predicted_reward_values = np.mean(reverse_reward_mapping_func(output_predicted_reward.cpu().numpy()),axis=0)

                    wandb.log({f"{self.polid}/all_predicted_reward": predicted_reward_values}, step=self.num_timesteps)
                    wandb.log({f"{self.polid}/all_rewards": all_rewards_values}, step=self.num_timesteps)
                    wandb.log({f"{self.polid}/reward_acc": reward_acc.mean()}, step=self.num_timesteps)
                    wandb.log({f"{self.polid}/reward_ood_rate": reward_ood_rate.mean()}, step=self.num_timesteps)
                    for polid in range(self.num_agents):
                        wandb.log({f"{self.polid}/predicted_reward/{polid}": predicted_reward_values[polid]}, step=self.num_timesteps)
                        wandb.log({f"{self.polid}/all_rewards/{polid}": all_rewards_values[polid]}, step=self.num_timesteps)
            if not reweighted_reward_losses == None:
                if reweighted_reward_losses != 0:
                    wandb.log({f"train/reweighted_reward_loss": reweighted_reward_losses.item()}, step=self.num_timesteps)




        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    # def sample_from_reweighted_list(self, obs:th.Tensor, action:th.Tensor, reweighted_list:list):

    def compute_cf_rewards(self,policy,all_last_obs,all_actions,polid,all_distributions,sample_number=10):
        all_cf_rewards = []

        all_last_obs = obs_as_tensor(np.array(all_last_obs), policy.device)

        # batch_size, num_agents, 1
        all_actions = obs_as_tensor(np.transpose(np.array(all_actions),(1,0,2)), policy.device)
        
        # extract obs features
        all_obs_features = []
        all_obs_features= [policy.policy.extract_features(all_last_obs[i]) for i in range(self.num_agents)]
        all_obs_features = th.stack(all_obs_features,dim=0).permute(1,0,2)

        # num_env, num_agent, obs_feat_size
        all_obs_features = all_obs_features.reshape(all_obs_features.shape[0],-1)
        all_obs_features = all_obs_features.repeat(sample_number, 1, 1).permute(1, 0, 2)

        # 1, batch_size, sample_size, 1
        cf_all_actions = copy.deepcopy(all_actions).squeeze(-1)
        cf_all_actions = cf_all_actions.unsqueeze(1).repeat(1,sample_number, 1) #.permute(1, 0, 2)

        for i in range(self.num_agents):
            if i != polid:
                # return not one-hot
                cf_action_i = self.generate_samples(all_distributions[i],sample_number).permute(1, 0)
                # cf_action_i = cf_action_i.unsqueeze(-1)

                # all_actions_index = th.cat((all_actions,cf_action_i),dim=1)
                cf_all_actions[:, :, i] = cf_action_i
        eye_matrix = th.eye(self.action_space.n,device=cf_all_actions.device)
        cf_all_actions = eye_matrix[cf_all_actions]

        # batch_size, sample_size, num_agents * num_action
        cf_all_actions = cf_all_actions.reshape(cf_all_actions.shape[0],cf_all_actions.shape[1],-1) #.permute(1,0,2)
        
        # batch_size, num_sample, obs_feat_size + num_agents * num_action
        all_obs_actions_features = th.cat((all_obs_features,cf_all_actions),dim=-1) #.permute(1,0,2)
        
        all_cf_rewards = policy.policy.reward_net(all_obs_actions_features,self.num_agents)[0]
        
        # argmax
        all_cf_rewards_class_index = th.argmax(all_cf_rewards,dim=-1).cpu().numpy()

        # Set reward not in the dict to be default excluded reward
        reverse_reward_mapping_func = np.frompyfunc(lambda key: REWARD_ENV_SPACE[self.env_name].get(key, OOD_INDEX[self.env_name][1]), 1, 1) #SPEED, Can the function be jit?
        all_cf_rewards_values = reverse_reward_mapping_func(all_cf_rewards_class_index)

        # average along sample dimension
        all_cf_rewards = np.mean(all_cf_rewards_values,axis=1)
        return all_cf_rewards

    def generate_samples(self,distribution,sample_number):
        return distribution.sample(th.Size([sample_number]))

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
