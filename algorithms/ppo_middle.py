from __future__ import absolute_import, division, print_function

from ray.rllib.agents.ppo.ppo import (
    choose_policy_optimizer,
    update_kl,
    validate_config,
    warn_about_bad_reward_scales,
)
from ray.rllib.agents.ppo.ppo_tf_policy import (
    KLCoeffMixin,
    ValueNetworkMixin,
    clip_gradients,
    kl_and_loss_stats,
    postprocess_ppo_gae,
    ppo_surrogate_loss,
    setup_config,
    setup_mixins,
    vf_preds_fetches,
)
from algorithms.common_funcs_middle import (
    middle_postprocess_trajectory,
    middle_fetches
)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy import build_tf_policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
import tensorflow as tf

from algorithms.common_funcs_baseline import BaselineResetConfigMixin

EXTRINSIC_REWARD = "extrinsic_reward"
TEAM_REWARD = "team_reward"

def extra_middle_model_stats(policy, train_batch):
    """
    Add stats that are logged in progress.csv
    :return: Combined reward model stats
    """
    base_stats = kl_and_loss_stats(policy, train_batch)
    base_stats = {
        **base_stats,
        "var_gnorm": tf.global_norm([x for x in policy.model.trainable_variables()]),
        EXTRINSIC_REWARD: tf.reduce_mean(train_batch[SampleBatch.REWARDS]),
        TEAM_REWARD: train_batch[TEAM_REWARD],
    }
    return base_stats

def postprocess_ppo_middle(policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Add the conterfactual reward to the trajectory.
    Then, add the policy logits, VF preds, and advantages to the trajectory.
    :return: Updated trajectory (batch)
    """
    batch = middle_postprocess_trajectory(sample_batch)
    batch = postprocess_ppo_gae(policy, batch)
    return batch


def extra_reward_fetches(policy):
    """
    Adds value function, logits, reward predictions to experience train_batch
    :return: Updated fetches
    """
    ppo_fetches = vf_preds_fetches(policy)
    ppo_fetches.update(middle_fetches(policy))
    return ppo_fetches

def build_ppo_middle_trainer(config):
    """
    Creates a PPO policy class, then creates a trainer with this policy.
    :param config: The configuration dictionary.
    :return: A new PPO trainer.
    """
    policy = build_tf_policy(
        name="PPOTFPolicy",
        get_default_config=lambda: config,
        loss_fn=ppo_surrogate_loss,
        stats_fn=extra_middle_model_stats,
        extra_action_fetches_fn=vf_preds_fetches,
        postprocess_fn=postprocess_ppo_middle,
        gradients_fn=clip_gradients,
        before_init=setup_config,
        before_loss_init=setup_mixins,
        mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin],
    )

    ppo_trainer = build_trainer(
        name="MiddlePPOTrainer",
        make_policy_optimizer=choose_policy_optimizer,
        default_policy=policy,
        default_config=config,
        validate_config=validate_config,
        after_optimizer_step=update_kl,
        after_train_result=warn_about_bad_reward_scales,
        mixins=[BaselineResetConfigMixin],
    )
    return ppo_trainer
