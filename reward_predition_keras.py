import tensorflow as tf
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Custom Mask Activation in TensorFlow
class MaskActivation(tf.keras.layers.Layer):
    def __init__(self, threshold=0.1, **kwargs):
        super(MaskActivation, self).__init__(**kwargs)
        self.threshold = threshold

    def call(self, x):
        mask = tf.cast(tf.abs(x) >= self.threshold, x.dtype)
        return x * mask



# Causal Model in TensorFlow
class CausalModel(tf.keras.Model):
    def __init__(self, input_dim, num_agents, enable_causality=True, dynamic_mask=True):
        super(CausalModel, self).__init__()
        self.enable_causality = enable_causality
        self.num_agents = num_agents
        self.dynamic_mask = dynamic_mask

        self.dense_layers = tf.keras.Sequential([
            Dense(1024, activation='relu'),
            Dense(512, activation='relu'),
            Dense(128, activation='relu'),
            Dense(1 if enable_causality else num_agents, activation='tanh')
        ])

        if self.enable_causality:
            if self.dynamic_mask:
                self.mask_predictor =  tf.keras.Sequential([
                    Dense(512, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(input_dim * num_agents, activation='sigmoid'),
                    MaskActivation()
                ])
            else:
                self.causal_mask = tf.Variable(tf.ones([num_agents, input_dim]), trainable=True)
                self.sh = 0.1

    def call(self, x, test=False):
        # x: [batch_size, input_dim]
        reg_loss = 0
        if self.enable_causality:
            # x: [batch_size, 1, input_dim]
            # causal_mask: [1, num_agent, input_dim]
            # masked_input: [batch_size, num_agent, input_dim]
            if self.dynamic_mask:
                mask = self.mask_predictor(x)
                mask = tf.reshape(mask, [-1, self.num_agents, x.shape[-1]])
                # reg_loss
                reg_loss = tf.reduce_mean(tf.abs(mask))
                # sparsity
                sparsity_tensor = tf.cast(tf.abs(mask) > self.sh, tf.float32)
                sparsity = tf.reduce_sum(sparsity_tensor) / tf.size(sparsity_tensor, out_type=tf.float32)      
            else:
                # if test:
                # get mask
                mask = tf.cast(tf.abs(self.causal_mask) > self.sh, tf.float32)
                mask *= self.causal_mask
                # mask = mask[tf.newaxis, :, :]
                mask = tf.expand_dims(mask, 0)

                # else:
                    # mask = self.causal_mask[tf.newaxis, :, :]
                    # mask = tf.expand_dims(self.causal_mask, 0)
                reg_loss = tf.reduce_mean(tf.abs(mask))
                sparsity_tensor = tf.cast(tf.abs(mask) > self.sh, tf.float32)
                sparsity = tf.reduce_sum(sparsity_tensor) / tf.size(sparsity_tensor, out_type=tf.float32)
            
            x = tf.expand_dims(x, 1) * mask
            # x = x[:, tf.newaxis, :] * mask

            x = self.dense_layers(x)
            # x = x[:, :, 0]
            x = tf.squeeze(x, -1)
            self.reg_loss = reg_loss
            self.sparsity = sparsity
            return x, reg_loss
        else:
            x = self.dense_layers(x)
            return x
    def get_reg_loss(self):
        return self.reg_loss

# Function to load and preprocess data
def load_data(file_path):
    obs_action = []
    rewards = []
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            obs_action.append(np.concatenate([data["vector_states"], data["actions"]]))
            rewards.append(data["rewards"])

    return np.array(obs_action, dtype=np.float32), np.array(rewards, dtype=np.float32)



# Load data
file_path = "./trajs_file.json"
obs_action, rewards = load_data(file_path)
enable_causality = True
dynamic_mask = False
# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(obs_action, rewards, test_size=0.2, random_state=42)

# Define and compile the model
input_dim = X_train.shape[1] # Adjust as per your data
num_agents = y_train.shape[1] # Adjust as per your data
model = CausalModel(input_dim, num_agents, enable_causality=enable_causality, dynamic_mask=dynamic_mask)
# model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())
# Assuming you have a model instance named 'model'
# Compile the model with the custom loss function
alpha = 1e-5    
loss_func = [MeanSquaredError()]
if enable_causality:
    loss_func += [lambda y_true, y_pred: y_pred * alpha]
model.compile(optimizer=Adam(learning_rate=0.01), loss = loss_func)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1024)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}")

# Save the model
model.save('reward_model_wo_causality.h5')
