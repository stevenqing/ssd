import sys

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from models.actor_critic_lstm import ActorCriticLSTM
from models.common_layers import build_conv_layers, build_fc_layers
from models.reward_lstm import rewardLSTM
from models.causal_reward_model import MaskActivation, CausalModel 
tf = try_import_tf()
from ray.rllib.models.tf.misc import normc_initializer

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

<<<<<<< HEAD
        self.moa_encoder_model = self.create_moa_encoder_model(obs_space, model_config)
        self.register_variables(self.moa_encoder_model.variables)
        self.moa_encoder_model.summary()

        # now output two heads, one for action selection and one for the prediction of other agents
        inner_obs_space = self.moa_encoder_model.output_shape[0][-1]
=======
        self.policy_model, self.reward_model = self.create_model(obs_space, model_config)
        self.register_variables(self.policy_model.variables + self.reward_model.variables)

        inner_obs_space = self.policy_model.output_shape[0]
>>>>>>> b82fe6df9bcfbc18347cc1b34bbaa66af7a763c4

        cell_size = model_config["custom_options"].get("cell_size")
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
            "train_reward_only_when_visible"
        ]
        self.conterfactual_only_when_visible = model_config["custom_options"][
            "conterfactual_only_when_visible"
        ]
        self.reward_weight = model_config["custom_options"]["reward_loss_weight"]


<<<<<<< HEAD
        self.moa_model = MoaLSTM(
            inner_obs_space,
            action_space,
            self.num_other_agents * num_outputs,
            model_config,
            "moa_model",
            cell_size=cell_size,
        )
        self.register_variables(self.actions_model.rnn_model.variables)
        self.register_variables(self.moa_model.rnn_model.variables)
        self.actions_model.rnn_model.summary()
        self.moa_model.rnn_model.summary()

    @staticmethod
    def create_moa_encoder_model(obs_space, model_config):
=======
        self.register_variables(self.actions_model.rnn_model.variables)
        self.register_variables(self.reward_model.variables)
        self.actions_model.rnn_model.summary()
        self.reward_model.summary()

    def create_model(self, obs_space, model_config):
>>>>>>> b82fe6df9bcfbc18347cc1b34bbaa66af7a763c4
        """
        Creates the convolutional part of the reward model.
        Also casts the input uint8 observations to float32 and normalizes them to the range [0,1].
        :param obs_space: The agent's observation space.
        :param model_config: The config dict containing parameters for the convolution type/shape.
        :return: A new Model object containing the convolution.
        """
        original_obs_dims = obs_space.original_space.spaces["curr_obs"].shape
        inputs = tf.keras.layers.Input(original_obs_dims, name="observations", dtype=tf.uint8)
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
        num_agents = action_dim = other_action_shape[0] + 1

        # causal mask 
        # causal_mask = tf.Variable(tf.ones([num_agents, vector_state_dim + action_dim]), trainable=True, name='causal_mask')
        # causal_mask =  tf.get_variable('causal_mask', [num_agents, vector_state_dim + action_dim], tf.float32,
        #                                      initializer=tf.compat.v1.ones_initializer()) # double check trainable


        # define inputs
        inputs_for_reward = tf.keras.layers.Input(vector_state_dim + action_dim, name="inputs_for_reward", dtype=tf.float32)
    
        # causal mask times input 
        # inputs_for_reward: [batch_size, vector_state_dim + action_dim] -> [batch_size, 1, vector_state_dim + action_dim]
        # causal_mask: [num_agents, vector_state_dim + action_dim] -> [1, num_agents, vector_state_dim + action_dim]
        # masked_input: [batch_size, num_agents, vector_state_dim + action_dim]
        causal_mask_layer = CAUSAL_MASK(input_dim=vector_state_dim + action_dim, num_agent=num_agents)
        masked_input = causal_mask_layer(inputs_for_reward)
        # masked_input = tf.reshape(inputs_for_reward, [-1, 1, inputs_for_reward.shape[1]]) * tf.reshape(causal_mask, [1, -1, inputs_for_reward.shape[1]])
        predicted_reward = self.get_reward_predictor(masked_input)
        predicted_reward = tf.squeeze(predicted_reward, axis=-1)

        return tf.keras.Model(inputs, [actor_critic_fc], name="Policy_Model"), tf.keras.Model(inputs_for_reward, predicted_reward, name="Reward_Predictor_Model")

    @staticmethod
    def get_reward_predictor(masked_input=None, emb_size=64):
        # layer 1
        last_layer = tf.keras.layers.Dense(
                emb_size,
                name="fc_{}_{}".format(1, 'reward'),
                activation='relu',
                kernel_initializer=normc_initializer(1.0),
            )(masked_input)
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
        return last_layer



    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        First evaluate non-LSTM parts of model. Then add a time dimension to the batch before
        sending inputs to forward_rnn(), which evaluates the LSTM parts of the model.
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: LSTM sequence lengths.
        :return: The agent's own action logits and the new model state.
        """
        # Evaluate non-lstm layers
        actor_critic_fc_output, reward_fc_output = self.reward_encoder_model(input_dict)
        # self.compute_causal_reward(input_dict, state[4], counterfactuals)
        #TODO: Not sure we should put the actor_critic_fc_output in here, maybe should be action_logits        
        return actor_critic_fc_output, reward_fc_output

    def compute_causal_reward(self, input_dict,  counterfactual_logits):
        """
        Compute causal_reward.
        :param input_dict: The model input tensors.
        """
        
        vector_state = input_dict["obs"]["vector_state"]
        agent_id = input_dict["obs"]["agent_id"]
        other_action = input_dict["obs"]["other_agent_actions"]
        actual_action = input_dict["obs"]["actions"]

        
        cf_action_list = []
        range_action = np.arange(0,action_range,1)
        if len(other_action) < 3:
            for i in range_action:
                for j in range_action:
                    cf_action_list.append([i,j])
        else:
        #TODO:to be implemented for cleanup or harvest(with larger action space and more agents; maybe use sampling method)
             pass 
	
        # Expand to (cf_dim, batch_size, action_number)
        B = np.shape(vector_state)[0]
        C = np.shape(cf_action_list)[0]	

        cf_action_list = tf.unsqueeze(tf.convert_to_tensor(cf_action_list),dim=1)
        cf_action_list = tf.repeat(cf_action_list, repeats=B, axis=1)

        vector_state = tf.unsqueeze(tf.convert_to_tensor(vector_state),dim=0)
        vector_state = tf.repeat(vector_state, repeats=C, axis=0)

        actual_action = tf.unsqueeze(tf.convert_to_tensor(actual_action),dim=0)
        actual_action = tf.reshape(tf.repeat(actual_action, repeats=C, axis=0), (C,B,-1))
        
        cf_action_total = tf.concat([cf_action_list[:,:,:agent_id[0]], actual_action, cf_action_list[:,:,agent_id[0]:]],axis=2)
        cf_vector_obs_action = tf.concat([vector_state,cf_action_total],axis=2)
        #TODO: Not sure how to implement the causal reward here
        causal_reward = REWARDModel()
        causal_reward = tf.reduce_sum(causal_reward, axis=-1)
        self._causal_reward = causal_reward

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

    def causal_reward(self):
        return self._causal_reward

    def predicted_rewards(self):
        """:returns Predicted rewards. NB: Since the agent's own true action is not known when
        evaluating this model, the timestep is off by one (too late). Thus, for any index n > 0,
        the value at n is a prediction made at n-1, about the actions taken at n.
        predicted_rewards[0] contains no sensible value, as this would have to be a prediction made
        at timestep -1, but we start time at 0."""
        return self._reward_pred

    def visibility(self):
        return tf.reshape(self._visibility, [-1, self.num_other_agents])

    def other_agent_actions(self):
        return tf.reshape(self._other_agent_actions, [-1, self.num_other_agents])

    @override(ModelV2)
    def get_initial_state(self):
        return self.actions_model.get_initial_state() + self.reward_model.get_initial_state()
