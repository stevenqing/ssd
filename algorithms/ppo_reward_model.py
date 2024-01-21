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

# introduce the trained model
# import torch
# save_path = '/scratch/prj/inf_du/shuqing/reward_model.pth'
# global reward_model
# reward_model = torch.load(save_path)
# reward_model.eval()

# TODO 统一reward model的接口
from algorithms.common_funcs_reward_model import (
    EXTRINSIC_REWARD,
    CONTERFACTUAL_REWARD,
    MAPING_REWARD_FROM_CLASS_TO_VALUE,
    MAPING_REWARD_FROM_VALUE_TO_CLASS,
    REWARDResetConfigMixin,
    build_model,
    get_reward_mixins,
    reward_fetches,
    reward_postprocess_trajectory,
    # setup_reward_model_loss,
    setup_reward_model_classification_loss,
    setup_reward_mixins,
    validate_reward_config,
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

    if model.discrete_rewards:
        reward_loss = setup_reward_model_classification_loss(model, train_batch)
        prediction_loss = reward_loss.classification_loss
    else:
        reward_loss = setup_reward_model_loss(model, train_batch)
        prediction_loss = reward_loss.mse_loss
    policy.reward_loss = prediction_loss
    if model.use_causal_mask:
        policy.reg_loss = reward_loss.reg_loss
    else:
        policy.reg_loss = tf.zeros([1])
    # some metric we want to focus
    # policy.correlation_factor = reward_loss.correlation_factor
    policy.pred_reward = reward_loss.pred_reward
    policy.true_reward = reward_loss.true_reward
    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)

    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        policy.entropy_coeff,
        policy.config["clip_param"],
        policy.config["vf_clip_param"],
        policy.config["vf_loss_coeff"],
        policy.config["use_gae"],
    ) 
    policy.loss_obj.loss += policy.reward_loss 
    policy.loss_obj.loss += policy.reg_loss

    return policy.loss_obj.loss

def extra_reward_fetches(policy):
    """
    Adds value function, logits, reward predictions to experience train_batch
    :return: Updated fetches
    """
    ppo_fetches = vf_preds_fetches(policy)
    ppo_fetches.update(reward_fetches(policy))
    return ppo_fetches




def extra_reward_model_stats(policy, train_batch):
    """
    Add stats that are logged in progress.csv
    :return: Combined reward model stats
    """
    base_stats = kl_and_loss_stats(policy, train_batch)
    base_stats = {
        **base_stats,
        "var_gnorm": tf.global_norm([x for x in policy.model.trainable_variables()]),
        # "cur_conterfactual_reward_weight": tf.cast(
        #     policy.cur_conterfactual_reward_weight_tensor, tf.float32
        # ),
        "reward_loss": policy.reward_loss,
        "reg_loss": policy.reg_loss,
        # "correlation_factor": policy.correlation_factor,
        "pred_reward": policy.pred_reward,
        "true_reward": policy.true_reward,
        EXTRINSIC_REWARD: tf.reduce_mean(train_batch[SampleBatch.REWARDS]),
        CONTERFACTUAL_REWARD: tf.reduce_mean(MAPING_REWARD_FROM_CLASS_TO_VALUE(train_batch[CONTERFACTUAL_REWARD])),
    }

    return base_stats


def postprocess_ppo_reward(policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Add the conterfactual reward to the trajectory.
    Then, add the policy logits, VF preds, and advantages to the trajectory.
    :return: Updated trajectory (batch)
    """
    batch = reward_postprocess_trajectory(policy,sample_batch)
    batch = postprocess_ppo_gae(policy, batch)
    return batch


def setup_ppo_reward_mixins(policy, obs_space, action_space, config):
    """
    Calls init on all PPO+MOA mixins in the policy
    """
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    setup_reward_mixins(policy, obs_space, action_space, config)


def validate_ppo_reward_config(config):
    """
    Validates the PPO+REWARD MODEL config
    :param config: The config to validate
    """
    validate_reward_config(config)
    validate_config(config)


def build_ppo_reward_trainer(reward_config):
    """
    Creates a MOA+PPO policy class, then creates a trainer with this policy.
    :param moa_config: The configuration dictionary.
    :return: A new MOA+PPO trainer.
    """
    tf.keras.backend.set_floatx("float32")

    trainer_name = "REWARDPPOTrainer"

    reward_ppo_policy = build_tf_policy(
        name="REWARDPPOTFPolicy",
        get_default_config=lambda: reward_config,
        loss_fn=loss_with_reward_model,
        make_model=build_model,
        stats_fn=extra_reward_model_stats,
        extra_action_fetches_fn=extra_reward_fetches,
        postprocess_fn=postprocess_ppo_reward,
        gradients_fn=clip_gradients,
        before_init=setup_config,
        before_loss_init=setup_ppo_reward_mixins,
        mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin]
        + get_reward_mixins(),
    )

    reward_ppo_trainer = build_trainer(
        name=trainer_name,
        default_policy=reward_ppo_policy,
        make_policy_optimizer=choose_policy_optimizer,
        default_config=reward_config,
        validate_config=validate_ppo_reward_config,
        after_optimizer_step=update_kl,
        after_train_result=warn_about_bad_reward_scales,
        mixins=[REWARDResetConfigMixin],
    )

    return reward_ppo_trainer
