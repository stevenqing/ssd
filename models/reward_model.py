import sys

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from models.actor_critic_lstm import ActorCriticLSTM
from models.common_layers import build_conv_layers, build_fc_layers
# from models.reward_lstm import rewardLSTM
from models.causal_reward_model import MaskActivation, CausalModel 
tf = try_import_tf()
from ray.rllib.models.tf.misc import normc_initializer
import numpy as np


class CAUSAL_MASK(tf.keras.layers.Layer):
    def __init__(self, input_dim=32,num_agent=32):
        super(CAUSAL_MASK, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(num_agent, input_dim), dtype="float32"),
            trainable=True,
        )
    def call(self, inputs, sh=0.1):
        mask = tf.where(tf.abs(self.w) > sh, self.w, tf.zeros_like(self.w))
        return tf.expand_dims(inputs, axis=1) * tf.expand_dims(mask, axis=0)
    def get_reg_loss(self, sh=0.1):
        mask = tf.where(tf.abs(self.w) > sh, tf.ones_like(self.w), tf.zeros_like(self.w))
        return tf.reduce_mean(mask)
    
class RewardModel(RecurrentTFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        A model with convolutional layers connected to two distinct sequences of fully connected
        layers. These then each connect to their own respective LSTM, one for an actor-critic policy,
        and one for modeling the actions of other agents (reward).
        :param obs_space: The agent's observation space.
        :param action_space: The agent's action space.
        :param num_outputs: The amount of actions available to the agent.
        :param model_config: The model config dict. Contains settings dictating layer sizes/amounts,
        amount of other agents, divergence measure used for social conterfactual, and other experiment
        parameters.
        :param name: The model name.
        """
        super(RewardModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_space = obs_space # (697, )

        # The inputs of the shared trunk. We will concatenate the observation space with
        # shared info about the visibility of agents.
        # Currently we assume all the agents have equally sized action spaces.
        self.num_outputs = num_outputs
        self.num_other_agents = model_config["custom_options"]["num_other_agents"]
        # self.conterfactual_divergence_measure = model_config["custom_options"][
        #     "conterfactual_divergence_measure"
        # ]

        # Declare variables that will later be used as loss fetches
        # It's
        self._model_out = None
        self._value_out = None
        self._action_pred = None
        self._counterfactuals = None
        self._other_agent_actions = None
        self._visibility = None
        self._social_conterfactual_reward = None
        self._true_one_hot_actions = None

        self.policy_model, self.reward_model = self.create_model(obs_space, model_config)
        self.register_variables(self.policy_model.variables + self.reward_model.variables)

        inner_obs_space = self.policy_model.output_shape[1]

        cell_size = model_config["custom_options"].get("cell_size")
        # print(inner_obs_space, action_space, num_outputs, model_config, cell_size)
        self.actions_model = ActorCriticLSTM(
            inner_obs_space,
            action_space,
            num_outputs,
            model_config,
            "action_logits",
            cell_size=cell_size,
        )

        # predicts the actions of all the agents besides itself
        # create a new input reader per worker
        self.train_reward_only_when_visible = model_config["custom_options"][
            "reward_only_when_visible"
        ]
        self.conterfactual_only_when_visible = model_config["custom_options"][
            "reward_only_when_visible"
        ]
        self.reward_loss_weight = model_config["custom_options"]["reward_loss_weight"]
        self.reg_loss_weight = model_config["custom_options"]["reg_loss_weight"]


        self.register_variables(self.actions_model.rnn_model.variables)
        self.register_variables(self.reward_model.variables)
        # self.actions_model.rnn_model.summary()
        # self.reward_model.summary()

    def create_model(self, obs_space, model_config):
        """
        Creates the convolutional part of the reward model.
        Also casts the input uint8 observations to float32 and normalizes them to the range [0,1].
        :param obs_space: The agent's observation space.
        :param model_config: The config dict containing parameters for the convolution type/shape.
        :return: A new Model object containing the convolution.
        """
        # agent_id_shape = obs_space.original_space["agent_id"].shape
        # (num_other_agents, action_range)
        other_action_shape = obs_space.original_space["other_agent_actions"].shape 
        # actual_action_shape = obs_space.original_space["actions"].shape

        # ==================== RGB encoder for policy ==================== #
        # RGB
        original_obs_dims = obs_space.original_space.spaces["curr_obs"].shape
        inputs = tf.keras.layers.Input(original_obs_dims, name="observations", dtype=tf.uint8)

        # Divide by 255 to transform [0,255] uint8 rgb pixel values to [0,1] float32.
        last_layer = tf.keras.backend.cast(inputs, tf.float32)
        last_layer = tf.math.divide(last_layer, 255.0)

        # Build the CNN layer
        conv_out = build_conv_layers(model_config, last_layer)

        # Build Actor-critic FC layers
        actor_critic_fc = build_fc_layers(model_config, conv_out, "policy")

        # ==================== vector encoder for reward model ==================== #
        # vector_state
        # joint obs space
        vector_state_dim = obs_space.original_space.spaces["vector_state"].shape[0]
        self.num_agents = num_agents = action_dim = other_action_shape[0] + 1

        # causal mask 
        # causal_mask = tf.Variable(tf.ones([num_agents, vector_state_dim + action_dim]), trainable=True, name='causal_mask')
        # causal_mask =  tf.get_variable('causal_mask', [num_agents, vector_state_dim + action_dim], tf.float32,
        #                                      initializer=tf.compat.v1.ones_initializer()) # double check trainable


        # define inputs
        inputs_for_reward = tf.keras.layers.Input(vector_state_dim + action_dim, name="inputs_for_reward", dtype=tf.float32)
        agent_id_matrix = tf.keras.layers.Input((num_agents, num_agents), name="inputs_id", dtype=tf.float32)

        # causal mask times input 
        # inputs_for_reward: [batch_size, vector_state_dim + action_dim] -> [batch_size, 1, vector_state_dim + action_dim]
        # causal_mask: [num_agents, vector_state_dim + action_dim] -> [1, num_agents, vector_state_dim + action_dim]
        # masked_input: [batch_size, num_agents, vector_state_dim + action_dim]
        self.causal_mask_layer = CAUSAL_MASK(input_dim=vector_state_dim + action_dim, num_agent=num_agents)
        masked_input = self.causal_mask_layer(inputs_for_reward)
        # print(masked_input)
        # masked_input = tf.concat([masked_input, agent_id_matrix], axis=-1)
        masked_input = tf.keras.layers.Concatenate(axis=-1)([masked_input, agent_id_matrix])
        predicted_reward = self.get_reward_predictor(masked_input,flag_discretize=True)
        predicted_reward = tf.squeeze(predicted_reward, axis=-1)

        return tf.keras.Model(inputs, [actor_critic_fc], name="Policy_Model"), tf.keras.Model([inputs_for_reward, agent_id_matrix], predicted_reward, name="Reward_Predictor_Model")

    @staticmethod
    def get_reward_predictor(masked_input_with_id, emb_size=64, flag_discretize=False, reward_range=5):
        # layer 1
        last_layer = tf.keras.layers.Dense(
                emb_size,
                name="fc_{}_{}".format(1, 'reward'),
                activation='relu',
                kernel_initializer=normc_initializer(1.0),
            )(masked_input_with_id)
        # layer 2
        last_layer = tf.keras.layers.Dense(
                    units=emb_size * 2, # output size for each agent's reward
                    name="fc_{}_{}".format(2, 'reward'),
                    activation='relu', # double check activation function, -4 to 4
                    kernel_initializer=normc_initializer(1.0),
                )(last_layer)
        # layer 3
        last_layer = tf.keras.layers.Dense(
                    units=emb_size, # output size for each agent's reward
                    name="fc_{}_{}".format(3, 'reward'),
                    activation='relu', # double check activation function, -4 to 4
                    kernel_initializer=normc_initializer(1.0),
                )(last_layer)
        # layer 4
        last_layer = tf.keras.layers.Dense(
                    units=1, # output size for each agent's reward
                    name="fc_{}_{}".format(4, 'reward'),
                    activation=None, # double check activation function, -4 to 4
                    kernel_initializer=normc_initializer(1.0),
                )(last_layer)
        if flag_discretize:
            softmax_output = tf.keras.layers.Dense(
                    units=reward_range, 
                    name="fc_{}_{}".format(4, 'softmax'),
                    activation='softmax',
                    kernel_initializer=normc_initializer(1.0),
                )(last_layer)
            last_layer = tf.math.argmax(softmax_output, axis=-1)
            last_layer = tf.expand_dims(last_layer,axis=-1)
            last_layer = last_layer-3
        return last_layer

    def get_reg_loss(self, ):
        return self._reg_loss
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens, training=True):
        """
        First evaluate non-LSTM parts of model. Then add a time dimension to the batch before
        sending inputs to forward_rnn(), which evaluates the LSTM parts of the model.
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens:: LSTM sequence lengths.
        :return: The agent's own action logits and the new model state.
        """
        # Evaluate non-lstm layers
        actor_critic_fc_output = self.policy_model(input_dict["obs"]["curr_obs"])
        rnn_input_dict = {
            "ac_trunk": actor_critic_fc_output,
            # "prev_reward_trunk": state[5],
            "other_agent_actions": input_dict["obs"]["other_agent_actions"],
            "visible_agents": input_dict["obs"]["visible_agents"],
            "prev_actions": input_dict["prev_actions"],
        }
        
        # Add time dimension to rnn inputs
        for k, v in rnn_input_dict.items():
            rnn_input_dict[k] = add_time_dimension(v, seq_lens)

        output, new_state = self.forward_rnn(rnn_input_dict, state, seq_lens)
        action_logits = tf.reshape(output, [-1, self.num_outputs])

        if training:
            # TODO using conterfactual reward & computing rewards
            # self._predicted_reward =  input_dict['prev_rewards'] 
            # self._counterfactual_rewards = input_dict['prev_rewards']
            self._predicted_reward = self.compute_reward(input_dict)
            self._counterfactual_rewards = self.compute_conterfactual_reward(input_dict)
            self._reg_loss = self.causal_mask_layer.get_reg_loss()
        return action_logits, new_state
    def forward_rnn(self, input_dict, state, seq_lens):
        """
        Forward pass through the MOA LSTMs.
        Implicitly assigns the value function output to self_value_out, and does not return this.
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: LSTM sequence lengths.
        :return: The policy logits and new LSTM states.
        """
        # Evaluate the actor-critic model
        pass_dict = {"curr_obs": input_dict["ac_trunk"]}
        h1, c1, *_ = state
        (
            self._model_out,
            self._value_out,
            output_h1,
            output_c1,
        ) = self.actions_model.forward_rnn(pass_dict, [h1, c1], seq_lens)        # cf_action_list = []
        # range_action = np.arange(0,int(other_action.shape[1]),1)
        # if other_action.shape[1] < 3:
        #     for i in range(action_range):
        #         for j in range(action_range):
        #             cf_action_list.append([i,j])
        # else:
        # #TODO:to be implemented for cleanup or harvest(with larger action space and more agents; maybe use sampling method)
        #      pass 

        # TODO(@evinitsky) move this into ppo_moa by using restore_original_dimensions()
        self._other_agent_actions = input_dict["other_agent_actions"]
        self._visibility = input_dict["visible_agents"]

        return self._model_out, [output_h1, output_c1]
    
    def compute_reward(self, input_dict):
        #TODO: Double check the input_dict['obs']['all_actions'](checked should be all right)
        states, actions = input_dict['obs']['prev_vector_state'], input_dict['obs']['all_actions']
        # agent_id_matrix = tf.transpose(agent_id_matrix,perm=[1,0,2])

        # states = tf.expand_dims(states,axis=0)
        # actions = tf.expand_dims(actions,axis=0)

        # all_vector_state = tf.repeat(states,agent_id_matrix.shape[-1],axis=0)
        # actions = tf.repeat(actions,agent_id_matrix.shape[-1],axis=0)

        # all_vector_state = tf.concat((all_vector_state[:,:,:negative_index],agent_id_matrix),axis=-1)

        # state_action = tf.concat((all_vector_state, actions), axis=-1)
        # if state_action.shape[1] > 0:
         #     state_action = tf.squeeze(state_action,axis=1)
        state_action = tf.concat((states, actions), axis=-1)

        predicted_reward = self.reward_model({'inputs_for_reward': state_action, 
                                                'inputs_id': input_dict['obs']['agent_id_matrix']})
        return predicted_reward

    def get_predicted_reward(self, ):
        return self._predicted_reward
    
    def get_conterfactual_reward(self, ):
        return self._counterfactual_rewards
    
    def compute_conterfactual_reward(self, input_dict, sample_number=10):
        """
        Compute counterfactual reward of other agents.
        :param input_dict: The model input tensors.
        :param prev_action_logits: Logits for the agent's own policy/actions at t-1
        :param counterfactual_logits: The counterfactual obs_action vector for actions made by other
        agents at t.
        """

        vector_state = input_dict["obs"]["vector_state"]
        agent_id_matrix = input_dict["obs"]["agent_id_matrix"]
        cf_actions = input_dict["obs"]["cf_actions"]

        cf_number = cf_actions.shape[1]
        agent_num = agent_id_matrix.shape[-1]
        
        # enabling the cf action 
        all_vector_state = tf.expand_dims(vector_state,axis=0)
        all_vector_state = tf.repeat(all_vector_state, cf_number, axis=0)

        cf_actions = tf.transpose(cf_actions,perm=[1,0,2])
        
        all_vector_state_action = tf.concat((all_vector_state,cf_actions),axis=-1)

        cf_agent_id_matrix = tf.expand_dims(agent_id_matrix,axis=0)
        cf_agent_id_matrix = tf.repeat(cf_agent_id_matrix,cf_number,axis=0)

        
        random_index = tf.random.uniform(shape=(sample_number,), maxval=cf_number, dtype=tf.int32)
        cf_vector_obs_action = tf.gather(all_vector_state_action, random_index)
        cf_agent_id_matrix = tf.gather(cf_agent_id_matrix,random_index)

        for i in range(sample_number):
            inputs_for_reward = cf_vector_obs_action[i]
            inputs_id = cf_agent_id_matrix[i]
            predicted_cf_reward = self.reward_model({'inputs_for_reward': inputs_for_reward, 
                                                'inputs_id': inputs_id})
            predicted_cf_reward = tf.expand_dims(predicted_cf_reward,axis=0)
            if i == 0:
                cf_reward = predicted_cf_reward
            else:
                cf_reward = tf.concat((cf_reward,predicted_cf_reward),axis=0)
            
        # cf_total_reward = self.reward_model({'inputs_for_reward': cf_vector_obs_action, 
        #                                         'inputs_id': cf_agent_id_matrix})
        cf_reward = tf.reduce_mean(cf_reward,axis=0)

        return cf_reward

    def marginalize_predictions_over_own_actions(self, prev_action_logits, counterfactual_logits):
        """
        Calculates marginal policies for all other agents.
        :param prev_action_logits: The agent's own policy logits at time t-1 .
        :param counterfactual_logits: The counterfactual action predictions made at time t-1 for
        other agents' actions at t.
        :return: The marginal policies for all other agents.
        """
        # Probability of each action in original trajectory
        logits = tf.nn.softmax(prev_action_logits)

        # Normalize to reduce numerical inaccuracies
        logits = logits / tf.reduce_sum(logits, axis=-1, keepdims=True)

        # Indexing is currently [B, Agent actions, num_other_agents * other_agent_logits]
        # Change to [B, Agent actions, num other agents, other agent logits]
        counterfactual_logits = tf.reshape(
            counterfactual_logits,
            [-1, self.num_outputs, self.num_other_agents, self.num_outputs],
        )

        counterfactual_logits = tf.nn.softmax(counterfactual_logits)
        # Change shape to broadcast probability of each action over counterfactual actions
        logits = tf.reshape(logits, [-1, self.num_outputs, 1, 1])
        normalized_counterfactual_logits = logits * counterfactual_logits
        # Remove counterfactual action dimension
        marginal_probs = tf.reduce_sum(normalized_counterfactual_logits, axis=-3)

        # Normalize to reduce numerical inaccuracies
        marginal_probs = marginal_probs / tf.reduce_sum(marginal_probs, axis=-1, keepdims=True)

        return marginal_probs

    @staticmethod
    def kl_div(x, y):
        """
        Calculate KL divergence between two distributions.
        :param x: A distribution
        :param y: A distribution
        :return: The KL-divergence between x and y. Returns zeros if the KL-divergence contains NaN
        or Infinity.
        """
        dist_x = tf.distributions.Categorical(probs=x)
        dist_y = tf.distributions.Categorical(probs=y)
        result = tf.distributions.kl_divergence(dist_x, dist_y)

        # Don't return nans or infs
        is_finite = tf.reduce_all(tf.is_finite(result))

        def true_fn():
            return result

        def false_fn():
            return tf.zeros(tf.shape(result))

        result = tf.cond(is_finite, true_fn=true_fn, false_fn=false_fn)
        return result

    def _reshaped_one_hot_actions(self, actions_tensor, name):
        """
        Converts the collection of all actions from a number encoding to a one-hot encoding.
        Then, flattens the one-hot encoding so that all concatenated one-hot vectors are the same
        dimension. E.g. with a num_outputs (action_space.n) of 3:
        _reshaped_one_hot_actions([0,1,2]) returns [1,0,0,0,1,0,0,0,1]
        :param actions_tensor: The tensor containing actions.
        :return: Tensor containing one-hot reshaped action values.
        """
        one_hot_actions = tf.keras.backend.one_hot(actions_tensor, self.num_outputs)
        # Extract partially known tensor shape and combine with actions_layer known shape
        # This combination is a bit contrived for a reason: the shape cannot be determined otherwise
        batch_time_dims = [
            tf.shape(one_hot_actions)[k] for k in range(one_hot_actions.shape.rank - 2)
        ]
        reshape_dims = batch_time_dims + [actions_tensor.shape[-1] * self.num_outputs]
        reshaped = tf.reshape(one_hot_actions, shape=reshape_dims, name=name)
        return reshaped

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def counterfactual_actions(self):
        return self._counterfactuals

    def action_logits(self):
        return self._model_out

    def social_conterfactual_reward(self):
        return self._causal_reward

    def predicted_rewards(self):
        """:returns Predicted rewards. NB: Since the agent's own true action is not known when
        evaluating this model, the timestep is off by one (too late). Thus, for any index n > 0,
        the value at n is a prediction made at n-1, about the actions taken at n.
        predicted_rewards[0] contains no sensible value, as this would have to be a prediction made
        at timestep -1, but we start time at 0."""
        #TODO: Need to be checked
        return self._causal_reward

    def visibility(self):
        return tf.reshape(self._visibility, [-1, self.num_other_agents])

    def other_agent_actions(self):
        return tf.reshape(self._other_agent_actions, [-1, self.num_other_agents])

    @override(ModelV2)
    def get_initial_state(self):
        return self.actions_model.get_initial_state() # + self.reward_model.get_initial_state()
