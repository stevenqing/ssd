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

# add by ReedZyd
PREDICTED_REWARD = "predicted_reward" # predicted rewards for reward model learning
CONTERFACTUAL_REWARD = "conterfactual_reward" # conterfactual rewards for policy learning
TRUE_REWARD = 'true_reward' # true rewards for reward model learning

def reward_postprocess_trajectory(sample_batch):
    # add conterfactual reward and add to batch.
    # TODO check if the timestep for reward can match the timestep for state
    # TODO add weight to the conterfactural reward
    conterfactual_reward = sample_batch[CONTERFACTUAL_REWARD]
    # print(sample_batch[CONTERFACTUAL_REWARD])
    sample_batch[EXTRINSIC_REWARD] = sample_batch["rewards"]
    sample_batch["rewards"] = sample_batch["rewards"] + np.sum(conterfactual_reward,axis=1)

    return sample_batch
            


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


