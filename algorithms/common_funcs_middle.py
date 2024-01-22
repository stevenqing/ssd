import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override
import torch
from algorithms.common_funcs_baseline import BaselineResetConfigMixin

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


PREDICTED_REWARD = "predicted_reward" # predicted rewards for reward model learning
CONTERFACTUAL_REWARD = "conterfactual_reward" # conterfactual rewards for policy learning
TRUE_REWARD = 'true_reward' # true rewards for reward model learning
TEAM_REWARD = 'team_reward'

def middle_postprocess_trajectory(sample_batch):
    # add conterfactual reward and add to batch.
    # TODO check if the timestep for reward can match the timestep for state
    # TODO add weight to the conterfactural reward
    team_reward = sample_batch[TEAM_REWARD]
    team_reward = tf.cast(team_reward, tf.int32)
    team_reward = tf.where(team_reward > 127, team_reward - 256, team_reward)
    # print(sample_batch[CONTERFACTUAL_REWARD])
    sample_batch[EXTRINSIC_REWARD] = sample_batch["rewards"]
    sample_batch["rewards"] = sample_batch["rewards"] + np.sum(team_reward,axis=1)

    return sample_batch
            


def middle_fetches(policy):
    """Adds logits, moa predictions of counterfactual actions to experience train_batches."""
    return {
        # Be aware that this is frozen here so that we don't
        # propagate agent actions through the reward
        # TODO(@evinitsky) remove this once we figure out how to split the obs
        ACTION_LOGITS: policy.model.action_logits(),
        OTHERS_ACTIONS: policy.model.other_agent_actions(), 
        VISIBILITY: policy.model.visibility(),
        TEAM_REWARD: policy.model.get_team_reward(),
    }


