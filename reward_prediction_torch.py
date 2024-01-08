import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import json
import numpy as np
from tqdm import tqdm

saving_path = './reward_model_wo_causality.pth'

# 1. Prepare the data
obs_action = []
rewards = []
obs = []
actions = []
with open("./trajs_file.json", "r") as file:
    for line in file:
        try:
            data = json.loads(line)
            obs.append(np.squeeze(np.array(data["vector_states"])))
            actions.append(np.squeeze(np.array(data["actions"])))
            rewards.append(np.squeeze(np.array(data["rewards"])).tolist())
            i = np.concatenate([np.squeeze(np.array(data["vector_states"])), np.squeeze(np.array(data["actions"]))], 0)
            i = i.tolist()
            if len(i) != 18:
                print(len(i))
            obs_action.append(i)
        except:
            pass

a = []
for i in range(len(obs_action)):
    a.append([obs_action[i]])
print(a[0])
print(np.shape(obs_action),np.shape(rewards))

# configure state dim, action dim, num_agents
input_dim = len(obs_action[0])
num_agent = len(rewards[0])
X_train, X_test, y_train, y_test = train_test_split(
    obs_action, rewards, test_size=0.2, random_state=42
)

X_train = np.reshape(X_train,[-1,18])
y_train = np.reshape(y_train,[-1,3])

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# 2. Data Standardization (if needed)

# 3. Build the model

class MaskActivation(nn.Module):
    def __init__(self, threshold=0.1):
        """
        Custom activation function where values near zero are set to zero.
        :param threshold: Values within [-threshold, threshold] are set to zero.
        """
        super(MaskActivation, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        x = torch.where(torch.abs(x) < self.threshold, torch.zeros_like(x), x)
        return x
class CausalModel(nn.Module):
    def __init__(self, input_dim, num_agents, enable_causality=True, dynamic_mask=True):
        super(CausalModel, self).__init__()
        self.enable_causality = enable_causality
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),  # Fully connected layer
            nn.ReLU(),
            nn.Linear(1024, 512),  # Fully connected layer
            nn.ReLU(),
            nn.Linear(512, 128),  # Fully connected layer
            nn.ReLU(),
            nn.Linear(128, 1 if enable_causality else num_agents),  # Output layer
            nn.Tanh() # TODO double check the activation function
            nn.Tanh() # TODO double check the activation function
        ) 
        if enable_causality:
            self.dynamic_mask = dynamic_mask
            if self.dynamic_mask:
                # input state, action
                self.mask_predictor = nn.Sequential(
                    nn.Linear(input_dim, 512),  # Fully connected layer
                    nn.ReLU(),
                    nn.Linear(512, 128),  # Fully connected layer
                    nn.ReLU(),
                    nn.Linear(128, input_dim * num_agent),  # Output layer
                    nn.Sigmoid(),
                    MaskActivation()
                )
            
            else:
                self.causal_mask = \
                    nn.Parameter(torch.ones(num_agent, input_dim), requires_grad=True)
                self.sh = 0.1

    def forward(self, x, test=False):
        # x: [batch_size, input_dim]
        if self.enable_causality:
            # x: [batch_size, 1, input_dim]
            # causal_mask: [1, num_agent, input_dim]
            # masked_input: [batch_size, num_agent, input_dim]
            if self.dynamic_mask:
                mask = self.mask_predictor(x).view(-1, num_agent, input_dim)
                reg_loss = mask.abs().mean()
                sparsity = mask.abs().nonzero().size(0) / mask.numel()
            else:
                if test:
                    mask = self.causal_mask.abs() > self.sh
                    mask *= self.causal_mask
                    mask = mask.unsqueeze(0)
                else:
                    mask = self.causal_mask.unsqueeze(0)
                reg_loss = self.get_reg_loss()
                sparsity = self.get_sparsity()
            input_ = x.unsqueeze(1) * mask
            # pred_rew: [batch_size, num_agent, 1]
            return self.layers(input_).unsqueeze(-1), reg_loss, sparsity

        else:
            input_ = x
            return self.layers(input_), 0, 0
    def get_reg_loss(self):
        return self.causal_mask.abs().mean()
    def get_sparsity(self):
        return (self.causal_mask.abs() > self.sh) / self.causal_mask.numel()
    
enable_causality = False
model = CausalModel(input_dim, num_agent, enable_causality=enable_causality)

# 4. Train the model
num_epochs = 10
batch_size = 1024
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
alpha = 1e-5
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for i in tqdm(range(0, len(X_train), batch_size), desc='Training', unit='batch', leave=True):
        inputs = X_train[i:i + batch_size]
        targets = y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs, reg_loss, sparsity = model(inputs)
        loss = criterion(outputs, targets) 
        if enable_causality and loss < 0.3:
            loss += alpha * reg_loss
        loss.backward()
        optimizer.step()
        
    print(f"{epoch}/{num_epochs} | Loss: {loss.item():.4f}, Sparsity: {sparsity:.3f}")

torch.save(model,saving_path)

# 5. Evaluate the model
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

with torch.no_grad():
    y_pred = model(X_test, test = True)[0]

    loss = criterion(y_pred, y_test)
    print(f"Test loss: {loss.item():.4f}")