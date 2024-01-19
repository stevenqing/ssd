import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override
import torch
from algorithms.common_funcs_baseline import BaselineResetConfigMixin
from scipy.stats import pearsonr
tf = try_import_tf()

MOA_PREDS = "moa_preds"
OTHERS_ACTIONS = "others_actions"
PREDICTED_ACTIONS = "predicted_actions"
VISIBILITY = "others_visibility"
VISIBILITY_MATRIX = "visibility_matrix"
EXTRINSIC_REWARD = "extrinsic_reward"

# Frozen logits of the policy that computed the action
ACTION_LOGITS = "action_logits"
COUNTERFACTUAL_ACTIONS = "counterfactual_actions"
POLICY_SCOPE = "func"

# add by ReedZyd
PREDICTED_REWARD = "predicted_reward" # predicted rewards for reward model learning
CONTERFACTUAL_REWARD = "conterfactual_reward" # conterfactual rewards for policy learning
TRUE_REWARD = 'true_reward' # true rewards for reward model learning
MAPING_REWARD_FROM_CLASS_TO_VALUE = lambda x: x - 4
MAPING_REWARD_FROM_VALUE_TO_CLASS = lambda x: x + 4
class InfluenceScheduleMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.baseline_influence_reward_weight = config["influence_reward_weight"]
        if any(
            config[key] is None
            for key in ["influence_reward_schedule_steps", "influence_reward_schedule_weights"]
        ):
            self.compute_influence_reward_weight = lambda: self.baseline_influence_reward_weight
        self.influence_reward_schedule_steps = config["influence_reward_schedule_steps"]
        self.influence_reward_schedule_weights = config["influence_reward_schedule_weights"]
        self.timestep = 0
        self.cur_influence_reward_weight = np.float32(self.compute_influence_reward_weight())
        # This tensor is for logging the weight to progress.csv
        self.cur_influence_reward_weight_tensor = tf.get_variable(
            "cur_influence_reward_weight",
            initializer=self.cur_influence_reward_weight,
            trainable=False,
        )

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(InfluenceScheduleMixIn, self).on_global_var_update(global_vars)
        self.timestep = global_vars["timestep"]
        self.cur_influence_reward_weight = self.compute_influence_reward_weight()
        self.cur_influence_reward_weight_tensor.load(
            self.cur_influence_reward_weight, session=self._sess
        )

    def compute_influence_reward_weight(self):
        """Computes multiplier for influence reward based on training steps
        taken and schedule parameters.
        """
        weight = np.interp(
            self.timestep,
            self.influence_reward_schedule_steps,
            self.influence_reward_schedule_weights,
        )
        return weight * self.baseline_influence_reward_weight
    

class REWARDLoss(object):
    def __init__(self, policy, reward_preds, true_rewards, loss_weight=1.0, others_visibility=None):
        """Train Reward prediction model with supervised cross entropy loss on a 
           trajectory.
           The model is trying to predict others' reward at timestep t+1 given all 
           states and actions at timestep t.
        Inputs:
            reward_preds: [B,N,1]
            true_rewards: [B,N,1]
        Returns:
            MSE loss
            reg loss
        """
        # Pred_logits[n] contains the prediction made at n-1 for actions taken at n, and a prediction
        # for t=0 cannot have been made at timestep -1, as the simulation starts at timestep 0.
        # Thus we remove the first prediction, as this value contains no sensible data.
        # NB: This means we start at n=1.

        # Compute MSE
        # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        # self.mse_per_entry = loss_fn(
        #                 true_rewards, tf.one_hot(reward_preds,depth=reward_preds.shape[-1]))
        # self.mse_per_entry = tf.losses.mean_squared_error(
        #                labels=true_rewards, predictions=reward_preds)
        mse = tf.keras.losses.MeanSquaredError()
        self.mse_per_entry = mse(
                        true_rewards, reward_preds)



        # Zero out the loss if the other agent isn't visible to this one.
        # print(true_rewards,reward_preds)
        # if others_visibility is not None:
        #     # others_visibility[n] contains agents visible at time n. We start at n=1,
        #     # so the first and last values have to be removed to maintain equal array size.
        #     others_visibility = others_visibility[1:-1, :]
        #     self.ce_per_entry *= tf.cast(others_visibility, tf.float32)

        # Flatten loss to one value for the entire batch
        self.mse_loss = tf.reduce_mean(self.mse_per_entry) * loss_weight[0]
        if policy.use_causal_mask:
            self.reg_loss = policy.get_reg_loss() * loss_weight[1]





class REWARDLossForClassification(object):
    def __init__(self, policy, reward_preds, true_rewards, loss_weight=1.0, others_visibility=None):
        """Train Reward prediction model with supervised cross entropy loss on a 
           trajectory.
           The model is trying to predict others' reward at timestep t+1 given all 
           states and actions at timestep t.
        Inputs:
            reward_preds: [B,N_agent,N_class]
            true_rewards: [B,N_agent]
        Returns:
            MSE loss
            reg loss
        """
        # Pred_logits[n] contains the prediction made at n-1 for actions taken at n, and a prediction
        # for t=0 cannot have been made at timestep -1, as the simulation starts at timestep 0.
        # Thus we remove the first prediction, as this value contains no sensible data.
        # NB: This means we start at n=1.

        classification_per_entry = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.mse_per_entry = classification_per_entry(
                        true_rewards, reward_preds)



        # Zero out the loss if the other agent isn't visible to this one.
        # print(true_rewards,reward_preds)
        # if others_visibility is not None:
        #     # others_visibility[n] contains agents visible at time n. We start at n=1,
        #     # so the first and last values have to be removed to maintain equal array size.
        #     others_visibility = others_visibility[1:-1, :]
        #     self.ce_per_entry *= tf.cast(others_visibility, tf.float32)

        # Flatten loss to one value for the entire batch
        self.classification_loss = tf.reduce_mean(self.mse_per_entry) * loss_weight[0]
        if policy.use_causal_mask:
            self.reg_loss = policy.get_reg_loss() * loss_weight[1]
        # correlation_reward_preds = tf.reduce_mean(reward_preds,axis=-1)
        # correlation_reward_preds = MAPING_REWARD_FROM_CLASS_TO_VALUE(correlation_reward_preds)
        # self.correlation_factor = self.get_rank_correlation(true_rewards,correlation_reward_preds)
        self.pred_reward = tf.argmax(reward_preds,-1)
        self.true_reward = true_rewards
        # tf.Print(self.mse_loss, [self.mse_loss], message="Reward MSE loss")
        # tf.Print(self.reg_loss, [self.reg_loss], message="Sparsity loss")
            

    def get_rank_correlation(self, x, y):
        y = tf.cast(y,tf.float32)
        x = tf.cast(x,tf.float32)
        mean_x = tf.reduce_mean(x, axis=0)
        mean_y = tf.reduce_mean(y, axis=0)
        # xm = x - mean_x
        # ym = y - mean_y
        r_num = tf.reduce_sum(mean_x * mean_y, axis=0)
        r_den = tf.sqrt(tf.reduce_sum(tf.square(mean_x), axis=0) * tf.reduce_sum(tf.square(mean_y), axis=0))
        r = r_num / (r_den + 1e-12)  
        return r


        # tf.Print(self.mse_loss, [self.mse_loss], message="Reward MSE loss")
        # tf.Print(self.reg_loss, [self.reg_loss], message="Sparsity loss")




def setup_reward_model_loss(policy, train_batch):
    # Instantiate the prediction loss
    reward_preds = train_batch[PREDICTED_REWARD] # need to reconsider in here, checking featches
    # true_rewards = train_batch[EXTRINSIC_REWARD]
    # true_rewards = train_batch['obs'][:,12:15]
    true_rewards = train_batch[TRUE_REWARD]
    true_rewards = tf.cast(true_rewards, tf.int32)
    true_rewards = tf.where(true_rewards > 127, true_rewards - 256, true_rewards)
    # 0/1 multiplier array representing whether each agent is visible to
    # the current agent.
    if policy.train_reward_only_when_visible:
        # if VISIBILITY in train_batch:
        others_visibility = train_batch[VISIBILITY]
    else:
        others_visibility = None
    # raise NotImplementedError
    reward_model_loss = REWARDLoss(
        policy, 
        reward_preds,
        true_rewards,
        loss_weight=[policy.reward_loss_weight, policy.reg_loss_weight], # not sure if it's good
        others_visibility=others_visibility,
    )
    return reward_model_loss


def setup_reward_model_classification_loss(policy, train_batch):
    # Instantiate the prediction loss
    reward_preds = train_batch[PREDICTED_REWARD] # need to reconsider in here, checking featches
    # true_rewards = train_batch[EXTRINSIC_REWARD]
    # true_rewards = train_batch['obs'][:,12:15]
    true_rewards = train_batch[TRUE_REWARD]
    true_rewards = tf.cast(true_rewards, tf.int32)
    true_rewards = tf.where(true_rewards > 127, true_rewards - 256, true_rewards)
    # map the reward from value to class
    true_rewards_class = MAPING_REWARD_FROM_VALUE_TO_CLASS(true_rewards)
    # 0/1 multiplier array representing whether each agent is visible to
    # the current agent.
    if policy.train_reward_only_when_visible:
        # if VISIBILITY in train_batch:
        others_visibility = train_batch[VISIBILITY]
    else:
        others_visibility = None
    # raise NotImplementedError
    reward_model_loss = REWARDLossForClassification(
        policy, 
        reward_preds,
        true_rewards_class,
        loss_weight=[policy.reward_loss_weight, policy.reg_loss_weight], # not sure if it's good
        others_visibility=others_visibility,
    )
    return reward_model_loss




def reward_postprocess_trajectory(policy,sample_batch):
    # add conterfactual reward and add to batch.
    # TODO check if the timestep for reward can match the timestep for statecorrelation
    # TODO add weight to the conterfactural reward
    cur_cf_reward_weight = policy.compute_influence_reward_weight()

    conterfactual_reward = sample_batch[CONTERFACTUAL_REWARD] 
    if policy.model.discrete_rewards:
        conterfactual_reward = MAPING_REWARD_FROM_CLASS_TO_VALUE(conterfactual_reward) # 1, sample_size, N_agents
        # mean along the first axis, which is the sample size
    conterfactual_reward = np.mean(conterfactual_reward,axis=1) # 1, N_agents
    # print(sample_batch[CONTERFACTUAL_REWARD])
    sample_batch[EXTRINSIC_REWARD] = sample_batch["rewards"]
    sample_batch["rewards"] = sample_batch["rewards"] + np.sum(conterfactual_reward,axis=1) * cur_cf_reward_weight

    return sample_batch
            
def agent_name_to_idx(agent_num, self_id):
    """split agent id around the index and return its appropriate position in terms
    of the other agents"""
    agent_num = int(agent_num)
    if agent_num > self_id:
        return agent_num - 1
    else:
        return agent_num


def get_agent_visibility_multiplier(trajectory, num_other_agents, agent_ids):
    traj_len = len(trajectory["obs"])
    visibility = np.zeros((traj_len, num_other_agents))
    for i, v in enumerate(trajectory[VISIBILITY]):
        vis_agents = [agent_name_to_idx(a, agent_ids[i]) for a in v]
        visibility[i, vis_agents] = 1
    return visibility




def reward_fetches(policy):
    """Adds logits, moa predictions of counterfactual actions to experience train_batches."""
    return {
        # Be aware that this is frozen here so that we don't
        # propagate agent actions through the reward
        # TODO(@evinitsky) remove this once we figure out how to split the obs
        ACTION_LOGITS: policy.model.action_logits(),
        OTHERS_ACTIONS: policy.model.other_agent_actions(), 
        VISIBILITY: policy.model.visibility(),
        PREDICTED_REWARD: policy.model.get_predicted_reward(),
        TRUE_REWARD: policy.model.get_true_reward(),
        CONTERFACTUAL_REWARD: policy.model.get_conterfactual_reward(), # check policy.model.predicted_actions()
    }


class REWARDConfigInitializerMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.num_other_agents = config["num_other_agents"]
        self.moa_loss_weight = tf.get_variable(
            "moa_loss_weight", initializer=config["moa_loss_weight"], trainable=False
        )
        self.influence_reward_clip = config["influence_reward_clip"]
        self.train_moa_only_when_visible = config["train_moa_only_when_visible"]
        self.influence_divergence_measure = config["influence_divergence_measure"]
        self.influence_only_when_visible = config["influence_only_when_visible"]


class REWARDResetConfigMixin(object):
    @staticmethod
    def reset_policies(policies, new_config, session):
        custom_options = new_config["model"]["custom_options"]
        for policy in policies:
            policy.moa_loss_weight.load(custom_options["moa_loss_weight"], session=session)
            policy.compute_influence_reward_weight = lambda: custom_options[
                "influence_reward_weight"
            ]

    def reset_config(self, new_config):
        policies = self.optimizer.policies.values()
        BaselineResetConfigMixin.reset_policies(policies, new_config)
        self.reset_policies(policies, new_config, self.optimizer.sess)
        self.config = new_config
        return True


def build_model(policy, obs_space, action_space, config):
    _, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])

    policy.model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        logit_dim,
        config["model"],
        name=POLICY_SCOPE,
        framework="tf",
    )

    return policy.model


def setup_reward_mixins(policy, obs_space, action_space, config):
    InfluenceScheduleMixIn.__init__(policy, config)
    REWARDConfigInitializerMixIn.__init__(policy, config)


def get_reward_mixins():
    return [
        REWARDConfigInitializerMixIn,
        InfluenceScheduleMixIn,
    ]


def validate_reward_config(config):
    config = config["model"]["custom_options"]
    if config["conterfactual_reward_weight"] < 0:
        raise ValueError("Conterfactual reward weight must be >= 0.")
