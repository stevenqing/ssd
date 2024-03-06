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
        features_dim=128,
        view_len=7,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
        enable_action_reward=False,
    ):
        super(CNN_Encoder, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        flat_out = num_frames * 12 * (view_len * 2 - 1) ** 2
        self.conv_1 = nn.Conv2d(
            in_channels=num_frames * 3,  # Input: (3 * 4) x 15 x 15
            out_channels=num_frames * 6,  # Output: 24 x 13 x 13
            kernel_size=3,
            stride=1,
            padding="valid",
        )
        self.conv_2 = nn.Conv2d(
            in_channels=num_frames * 6,  # Input: 24 x 13 x 13
            out_channels=num_frames * 12,  # Output: 48 x 11 x 11
            kernel_size=3,
            stride=1,
            padding="valid",
        )

        self.fc = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.mu_header = nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])
        self.sigma_header = nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

    def forward(self, observations) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        observations = observations.permute(0, 3, 1, 2)
        observations = F.relu(self.conv_1(observations))
        features = torch.flatten(F.relu(self.conv_2(observations)), start_dim=1)

        features = self.fc(features)
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
        enable_action_reward=False,
    ):
        super(CNN_Decoder, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        flat_out = num_frames * 12 * (view_len * 2 - 1) ** 2
        self.fc1 = nn.Linear(in_features=fcnet_hiddens[1], out_features=flat_out)
        self.conv_1 = nn.ConvTranspose2d(
            in_channels=num_frames * 12,  # Input: (3 * 4) x 15 x 15
            out_channels=num_frames * 6,  # Output: 24 x 13 x 13
            kernel_size=3,
            stride=1,
            padding="valid",
        )
        self.conv_2 = nn.ConvTranspose2d(
            in_channels=num_frames * 6,  # Input: 24 x 13 x 13
            out_channels=num_frames * 3,  # Output: 48 x 11 x 11
            kernel_size=3,
            stride=1,
            padding="valid",
        )


    def forward(self, features) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        features = self.fc1(features)
        features = features.unsqueeze(-1).unsqueeze(-1)
        features = F.relu(self.conv_1(features))
        recon = F.sigmoid(self.conv_2(features))

        return recon

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = CNN_Encoder
        self.decoder = CNN_Decoder
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        
        recon = self.decoder(z)
        return recon, mu, sigma

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return mu + eps * std