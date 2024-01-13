import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Custom Mask Activation in TensorFlow
class MaskActivation(tf.keras.layers.Layer):
    def __init__(self, threshold=0.1, **kwargs):
        super(MaskActivation, self).__init__(**kwargs)
        self.threshold = threshold

    def forward(self, x):
        mask = tf.cast(tf.abs(x) >= self.threshold, x.dtype)
        return x * mask

# Causal Model in TensorFlow
class CausalModel(tf.keras.Model):
    def __init__(self, input_dim, num_agents, enable_causality=True, dynamic_mask=True):
        super(CausalModel, self).__init__()
        self.enable_causality = enable_causality
        self.num_agents = num_agents
        self.dynamic_mask = dynamic_mask

        self.dense_layers = [
            Dense(1024, activation='relu'),
            Dense(512, activation='relu'),
            Dense(128, activation='relu'),
            Dense(1 if enable_causality else num_agents, activation='tanh')
        ]

        if enable_causality and dynamic_mask:
            self.mask_predictor = [
                Dense(512, activation='relu'),
                Dense(128, activation='relu'),
                Dense(input_dim * num_agents, activation='sigmoid'),
                MaskActivation()
            ]

    def forward(self, inputs, training=False):
        x = inputs
        if self.enable_causality and self.dynamic_mask:
            mask = x
            for layer in self.mask_predictor:
                mask = layer(mask)
            mask = tf.reshape(mask, [-1, self.num_agents, x.shape[-1]])
            x = x[:, tf.newaxis, :] * mask

        for layer in self.dense_layers:
            x = layer(x)
        return x
    def get_reg_loss(self):
	    return tf.reduced_mean(tf.abs(self.causal_mask))
    
    def get_sparsity(self):
        # Implement your method to calculate sparsity if needed
	    return (tf.abs(self.causal_mask) > self.sh) / tf.size(self.causal_mask)        

