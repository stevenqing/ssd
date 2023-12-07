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
    postprocess_ppo_gae,
    setup_config,
)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf

from algorithms.common_funcs_moa import build_model, get_moa_mixins, moa_postprocess_trajectory
from algorithms.common_funcs_scm import (
    SOCIAL_CURIOSITY_REWARD,
    SCMResetConfigMixin,
    get_curiosity_mixins,
    scm_fetches,
    scm_postprocess_trajectory,
    setup_scm_loss,
    setup_scm_mixins,
    validate_scm_config,
)
from algorithms.ppo_moa import (
    extra_moa_fetches,
    extra_moa_stats,
    loss_with_moa,
    setup_ppo_moa_mixins,
    validate_moa_config,
)

tf = try_import_tf()


def loss_with_scm(policy, model, dist_class, train_batch):
    """
    Calculate PPO loss with SCM and MOA loss
    :return: Combined PPO+MOA+SCM loss
    """
    _ = loss_with_moa(policy, model, dist_class, train_batch)

    selfish_loss = setup_selfish_loss(policy, train_batch)
    policy.selfish_loss = selfish_loss.total_loss

    # PPO loss_obj has already been instantiated in loss_with_moa
    policy.loss_obj.loss += selfish_loss.total_loss
    return policy.loss_obj.loss


def extra_scm_fetches(policy):
    """
    Adds value function, logits, moa predictions, SCM loss/reward to experience train_batches.
    :return: Updated fetches
    """
    ppo_fetches = selfish_fetches(policy))
    return ppo_fetches


def extra_selfish_stats(policy, train_batch):
    """
    Add stats that are logged in progress.csv
    :return: Combined PPO+MOA+SCM stats
    """
    selfish_stats = {
        "reward_weight_for_total_reward": tf.cast(
            policy.selfish_reward_weight_tensor, tf.float32
        ),
        # \alpha * R_total
        TOTAL_WEIGHTED_REWARD: train_batch[SELFISH_WEIGHTED_REWARD],
        "selfish_loss": policy.selfish_loss,
    }
    return selfish_stats


def postprocess_ppo_selfish(policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Add the influence and curiosity reward to the trajectory.
    Then, add the policy logits, VF preds, and advantages to the trajectory.
    :return: Updated trajectory (batch)
    """
    batch = selfish_postprocess_trajectory(policy, batch)
    batch = postprocess_ppo_gae(policy, batch)
    return batch


def setup_ppo_selfish_mixins(policy, obs_space, action_space, config):
    """
    Calls init on all PPO+MOA+SCM mixins in the policy
    """
    setup_selfish_mixins(policy, obs_space, action_space, config)


def validate_ppo_scm_config(config):
    """
    Validates the PPO+MOA+SCM config
    :param config: The config to validate
    """
    validate_scm_config(config)
    validate_moa_config(config)
    validate_config(config)


def build_ppo_scm_trainer(scm_config):
    """
    Creates a SCM+MOA+PPO policy class, then creates a trainer with this policy.
    :param scm_config: The configuration dictionary.
    :return: A new SCM+MOA+PPO trainer.
    """
    tf.keras.backend.set_floatx("float32")

    trainer_name = "SCMPPOTrainer"

    scm_ppo_policy = build_tf_policy(
        name="SCMPPOTFPolicy",
        get_default_config=lambda: scm_config,
        loss_fn=loss_with_scm,
        make_model=build_model,
        stats_fn=extra_scm_stats,
        extra_action_fetches_fn=extra_scm_fetches,
        postprocess_fn=postprocess_ppo_scm,
        gradients_fn=clip_gradients,
        before_init=setup_config,
        before_loss_init=setup_ppo_scm_mixins,
        mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin]
        + get_moa_mixins()
        + get_curiosity_mixins(),
    )

    scm_ppo_trainer = build_trainer(
        name=trainer_name,
        default_policy=scm_ppo_policy,
        make_policy_optimizer=choose_policy_optimizer,
        default_config=scm_config,
        validate_config=validate_ppo_scm_config,
        after_optimizer_step=update_kl,
        after_train_result=warn_about_bad_reward_scales,
        mixins=[SCMResetConfigMixin],
    )
    return scm_ppo_trainer
