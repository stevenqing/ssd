import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import json
import numpy as np
from tqdm import tqdm

saving_path = './reward_model.pth'

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

X_train, X_test, y_train, y_test = train_test_split(
    obs_action, rewards, test_size=0.2, random_state=42
)

X_train = np.reshape(X_train,[-1,18])
y_train = np.reshape(y_train,[-1,3])

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# 2. Data Standardization (if needed)

# 3. Build the model
model = nn.Sequential(
    nn.Flatten(),  # Flatten the input
    nn.Linear(len(obs_action[1]), 1024),  # Fully connected layer
    nn.ReLU(),
    nn.Linear(1024, 512),  # Fully connected layer
    nn.ReLU(),
    nn.Linear(512, 128),  # Fully connected layer
    nn.ReLU(),
    nn.Linear(128, len(rewards[0])),  # Output layer
    nn.Sigmoid()
)

causal_mask = nn.Parameter(torch.ones(len(obs_action[1]), len(rewards[0])), requires_grad=True)

# 4. Train the model
num_epochs = 1
batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for i in tqdm(range(0, len(X_train), batch_size), desc='Training', unit='batch', leave=True):
        inputs = X_train[i:i + batch_size]
        targets = y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

torch.save(model,saving_path)

# 5. Evaluate the model
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
