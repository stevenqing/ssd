from ray.rllib.agents.ppo.ppo import (
    choose_policy_optimizer,
    update_kl,
    validate_config,
    warn_about_bad_reward_scales,
)
from ray.rllib.agents.ppo.ppo_tf_policy import (
    KLCoeffMixin,
    PPOLoss,
    ValueNetworkMixin,
    clip_gradients,
    kl_and_loss_stats,
    postprocess_ppo_gae,
    setup_config,
    vf_preds_fetches,
)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf

from algorithms.common_funcs_reward_model import (
    EXTRINSIC_REWARD,
    PREDICTED_REWARD,
    REWARDResetConfigMixin,
    build_model,
    get_reward_model_mixins,
    reward_model_fetches,
    reward_model_postprocess_trajectory,
    setup_reward_model_loss,
    setup_reward_model_mixins,
    validate_reward_model_config,
)

tf = try_import_tf()

POLICY_SCOPE = "func"


def loss_with_reward_model(policy, model, dist_class, train_batch):
    """
    Calculate reward model loss
    :return: Reward model loss
    """
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    reward_model_loss = setup_reward_model_loss(logits, policy, train_batch)
    policy.reward_model_loss = reward_model_loss.total_loss 
    return policy.reward_model_loss

def extra_reward_model_stats(policy, train_batch):
    """
    Add stats that are logged in progress.csv
    :return: Combined reward model stats
    """
    base_stats = kl_and_loss_stats(policy, train_batch)
    base_stats = {
        **base_stats,
        "var_gnorm": tf.global_norm([x for x in policy.model.trainable_variables()]),
        "moa_loss": policy.reward_model_loss,
    }

    return base_stats


def postprocess_ppo_moa(policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Add the influence reward to the trajectory.
    Then, add the policy logits, VF preds, and advantages to the trajectory.
    :return: Updated trajectory (batch)
    """
    batch = moa_postprocess_trajectory(policy, sample_batch)
    batch = postprocess_ppo_gae(policy, batch)
    return batch


def setup_ppo_moa_mixins(policy, obs_space, action_space, config):
    """
    Calls init on all PPO+MOA mixins in the policy
    """
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    setup_moa_mixins(policy, obs_space, action_space, config)


def validate_ppo_moa_config(config):
    """
    Validates the PPO+MOA config
    :param config: The config to validate
    """
    validate_moa_config(config)
    validate_config(config)


def build_ppo_moa_trainer(moa_config):
    """
    Creates a MOA+PPO policy class, then creates a trainer with this policy.
    :param moa_config: The configuration dictionary.
    :return: A new MOA+PPO trainer.
    """
    tf.keras.backend.set_floatx("float32")

    trainer_name = "MOAPPOTrainer"

    moa_ppo_policy = build_tf_policy(
        name="MOAPPOTFPolicy",
        get_default_config=lambda: moa_config,
        loss_fn=loss_with_moa,
        make_model=build_model,
        stats_fn=extra_moa_stats,
        extra_action_fetches_fn=extra_moa_fetches,
        postprocess_fn=postprocess_ppo_moa,
        gradients_fn=clip_gradients,
        before_init=setup_config,
        before_loss_init=setup_ppo_moa_mixins,
        mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin]
        + get_moa_mixins(),
    )

    moa_ppo_trainer = build_trainer(
        name=trainer_name,
        default_policy=moa_ppo_policy,
        make_policy_optimizer=choose_policy_optimizer,
        default_config=moa_config,
        validate_config=validate_ppo_moa_config,
        after_optimizer_step=update_kl,
        after_train_result=warn_about_bad_reward_scales,
        mixins=[MOAResetConfigMixin],
    )

    return moa_ppo_trainer