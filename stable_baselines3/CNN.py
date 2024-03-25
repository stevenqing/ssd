from torch import nn
import numpy as np
import gym
import torch
import torch.nn.functional as F

class CNN_Encoder(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        reward_dim=5,
        action_dim=4*10,
        features_dim=128,
        view_len=7,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
        num_agents=5,
        enable_action_reward=False,
    ):
        super(CNN_Encoder, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        flat_out = num_frames * num_agents * 3 * (view_len * 2 - 3) ** 2
        self.conv_1 = nn.Conv2d(
            in_channels=num_frames * 3 * num_agents,  # Input: (3 * 4) x 15 x 15
            out_channels=num_frames * 6 * num_agents,  # Output: 24 x 13 x 13
            kernel_size=3,
            stride=1,
            padding="valid",
        )
        self.conv_2 = nn.Conv2d(
            in_channels=num_frames * 6 * num_agents,  # Input: 24 x 13 x 13
            out_channels=num_frames * 3 * num_agents,  # Output: 48 x 11 x 11
            kernel_size=3,
            stride=1,
            padding="valid",
        )
        self.reward_layer = nn.Linear(in_features=num_agents, out_features=features_dim)
        self.action_layer = nn.Linear(in_features=action_dim, out_features=features_dim)
        self.fc = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])

        self.mu_header = nn.Linear(in_features=fcnet_hiddens[0] + features_dim*2, out_features=fcnet_hiddens[1])
        self.sigma_header = nn.Linear(in_features=fcnet_hiddens[0] + features_dim*2, out_features=fcnet_hiddens[1])

    def forward(self, observations, action, reward) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        if observations.shape[1] != 90:
            observations = observations.permute(0, 3, 1, 2)
        observations = F.relu(self.conv_1(observations.float()/255.0))
        features = torch.flatten(F.relu(self.conv_2(observations)), start_dim=1)
        obs_features = self.fc(features)
        if action.dtype != torch.float32:
            action = action.float()
            reward = reward.float()
        action_features = F.relu(self.action_layer(action))
        reward_features = F.relu(self.reward_layer(reward))
        features = torch.cat([obs_features, action_features, reward_features], dim=-1)
        mu = self.mu_header(features)
        sigma = self.sigma_header(features)

        return mu, sigma

class CNN_Decoder(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim=128,
        view_len=7,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
        num_agents=3,
        enable_action_reward=False,
    ):
        super(CNN_Decoder, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        flat_out = num_frames * num_agents * 3 * (view_len * 2 - 3) ** 2
        self.view_len = view_len
        self.num_frames = num_frames
        self.num_agents = num_agents
        self.fc1 = nn.Linear(in_features=fcnet_hiddens[1], out_features=flat_out)
        self.conv_1 = nn.ConvTranspose2d(
            in_channels=num_frames * 3 * num_agents,  # Input: (3 * 4) x 15 x 15
            out_channels=num_frames * 6 * num_agents,  # Output: 24 x 13 x 13
            kernel_size=3,
            stride=1,
            # padding="valid",
        )
        self.conv_2 = nn.ConvTranspose2d(
            in_channels=num_frames * 6 * num_agents,  # Input: 24 x 13 x 13
            out_channels=num_frames * 3 * num_agents,  # Output: 48 x 11 x 11
            kernel_size=3,
            stride=1,
            # padding="valid",
        )


    def forward(self, features) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        features = self.fc1(features)
        features = features.unsqueeze(-1).unsqueeze(-1)
        features = features.view(-1, self.num_frames*3*self.num_agents, self.view_len*2-3, self.view_len*2-3)
        features = F.relu(self.conv_1(features))
        recon = F.sigmoid(self.conv_2(features))

        return recon

class VAE(nn.Module):
    def __init__(self,
        observation_space: gym.spaces.Box,
        reward_dim=3,
        action_dim=4*10,
        features_dim=128,
        view_len=7,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
        num_agents=5,
        enable_action_reward=False,):
        super(VAE, self).__init__()
        self.encoder = CNN_Encoder(observation_space, reward_dim, action_dim, features_dim, view_len, num_frames, fcnet_hiddens, num_agents, enable_action_reward)
        self.decoder = CNN_Decoder(observation_space, features_dim, view_len, num_frames, fcnet_hiddens, num_agents, enable_action_reward)
    
    def forward(self, obs, action, reward):
        mu, sigma = self.encoder(obs, action, reward)
        z = self.reparameterize(mu, sigma)
        
        recon = self.decoder(z)
        return recon, mu, sigma

    def encode(self, obs, action, reward):
        if action.dtype != reward.dtype:
            action = action.float()
            reward = reward.float()
        mu, sigma = self.encoder(obs, action, reward)
        return mu, sigma

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, recon_x, x, mu, logvar):
        x = x.permute(0, 3, 1, 2)
        BCE = F.mse_loss(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD