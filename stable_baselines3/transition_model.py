import argparse
import copy
import random
import time

from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal
from torch.utils.data import DataLoader, TensorDataset

####### Decoder #######
# the input of the decoder should be the latent variable z and the action
class P_or_za(nn.Module):
    def __init__(self, dim_z=128, nh=2, dim_h=128, dim_out_o=128*5, dim_out_r=5):
        super().__init__()
        self.nh = nh

        # Hidden layers
        self.hidden_z = nn.Sequential(nn.Linear(dim_z, dim_h), nn.ReLU())

        self.output_o = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())
                        for _ in range(nh)
                    ],
                    nn.Linear(dim_h, dim_out_o),
                )
            ]
        )
        
        self.output_r = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())
                        for _ in range(nh)
                    ],
                    nn.Linear(dim_h, dim_out_r),
                )
            ]
        )

    def forward(self, z, a):
        z = self.hidden_z(z).squeeze(1)
        o = torch.stack([o(z) for o in self.output_o]).swapaxes(0, 1).squeeze(1)
        r = torch.stack([r(z) for r in self.output_r]).swapaxes(0, 1).squeeze(1)
        return o, r





####### Encoder #######
# the input of the encoder should be the state, observation, action, reward
# the output of the encoder should be the latent variable z

# class Q_z_oar(nn.Module):
#     def __init__(self, dim_s=128, dim_in_oa=128*5+8*5, dim_r=5, nh=2, dim_h=128, dim_out=128):
#         super().__init__()
#         self.nh = nh
#         self.dim_out = dim_out
#         # Shared layers with separated output layers
#         self.hidden_state = nn.Sequential(nn.Linear(dim_s, dim_h), nn.ReLU())
#         self.hidden_oa = nn.Sequential(nn.Linear(dim_in_oa, dim_h), nn.ReLU())
#         self.hidden_r = nn.Sequential(nn.Linear(dim_r, dim_h), nn.ReLU())

#         self.hidden_layer = nn.Sequential(nn.Linear(dim_h*3, dim_h), nn.ReLU())

#         self.mu_header = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     *[
#                         nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())
#                         for _ in range(nh)
#                     ],
#                     nn.Linear(dim_h, dim_out),
#                 )
#             ]
#         )
#         self.sigma_header = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     *[
#                         nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())
#                         for _ in range(nh)
#                     ],
#                     nn.Linear(dim_h, dim_out),
#                 )
#             ]
#         )


#     def forward(self, s, oa, r) -> Normal:
#         s = self.hidden_state(s)
#         oa = self.hidden_oa(oa)
#         r = self.hidden_r(r)
#         x = torch.cat([s, oa, r], -1)
#         x = self.hidden_layer(x)

#         mu = torch.stack([mu(x) for mu in self.mu_header]).swapaxes(0, 1)
#         sigma = torch.stack([sigma(x) for sigma in self.sigma_header]).swapaxes(0, 1)
#         z = Normal(mu, torch.exp(sigma))
#         return z

class Q_z_oar(nn.Module):
    def __init__(self, dim_in_oa=128*5+8*5, dim_r=5, nh=2, dim_h=128, dim_out=128):
        super().__init__()
        self.nh = nh
        self.dim_out = dim_out
        # Shared layers with separated output layers
        self.hidden_oa = nn.Sequential(nn.Linear(dim_in_oa, dim_h), nn.ReLU())
        self.hidden_r = nn.Sequential(nn.Linear(dim_r, dim_h), nn.ReLU())

        self.hidden_layer = nn.Sequential(nn.Linear(dim_h*2, dim_h), nn.ReLU())

        self.mu_header = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())
                        for _ in range(nh)
                    ],
                    nn.Linear(dim_h, dim_out),
                )
            ]
        )
        self.sigma_header = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())
                        for _ in range(nh)
                    ],
                    nn.Linear(dim_h, dim_out),
                )
            ]
        )


    def forward(self, oa, r) -> Normal:
        oa = self.hidden_oa(oa)
        r = self.hidden_r(r)
        x = torch.cat([oa, r], -1)
        x = self.hidden_layer(x)

        mu = torch.stack([mu(x) for mu in self.mu_header]).swapaxes(0, 1)
        sigma = torch.stack([sigma(x) for sigma in self.sigma_header]).swapaxes(0, 1)
        z = Normal(mu, torch.exp(sigma))
        return z
    
class Transition_VAE(nn.Module):
    def __init__(self, hidden_size, dim_o, dim_a, dim_r, dim_z): # dim_s = dim_z
        super().__init__()
        nh=2
        self.encoder = Q_z_oar(dim_in_oa=dim_o+dim_a, dim_r=dim_r, nh=nh, dim_h=hidden_size, dim_out=dim_z)
        self.decoder = P_or_za(dim_z=dim_z,dim_out_r=dim_r,nh=nh, dim_h=hidden_size, dim_out_o=dim_o)

    def forward(self, s, o, a, r):
        oa = torch.cat([o, a], -1)
        z_dist = self.encoder(s, oa, r)
        z = z_dist.rsample()
        o_predicted,r_predicted = self.decoder(z)
        return o_predicted, r_predicted

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, o, a, r):
        # Compute z distribution
        oa = torch.cat([o, a], -1)
        z_dist = self.encoder(oa, r)
        z = z_dist.rsample()

        # Compute prediction distribution
        o_predicted,r_predicted = self.decoder(z, a)

        # Calculate the loss
        loss_o_recon = F.mse_loss(o, o_predicted)
        loss_r_recon = F.mse_loss(r, r_predicted)
        loss_recon = loss_o_recon + loss_r_recon

        # Compute total loss
        kl_divergence = -0.5 * (1 + 2 * z_dist.scale.log() - z_dist.mean.pow(2) - z_dist.scale.pow(2)).sum(-1).mean()
        loss = loss_recon + kl_divergence
        return loss, loss_recon, kl_divergence

class Transition_Net(nn.Module):
    def __init__(self, hidden_size, dim_latent_state, dim_a, dim_output_size):
        super().__init__()
        self.latent_state_input = nn.Linear(dim_latent_state, hidden_size)
        self.action_input = nn.Linear(dim_a, hidden_size)
        self.transition = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dim_output_size)
        )
    def forward(self, latent_state, action):
        if latent_state.dtype != action.dtype:
            action = action.to(latent_state.dtype)
        z = torch.cat([self.latent_state_input(latent_state), self.action_input(action)], -1)
        return self.transition(z)
# s_1 = torch.rand(32,2, 128)
# r = torch.rand(32,2, 5)
# o = torch.rand(32,2, 128*5)
# a = torch.rand(32,2, 8*5)
# model = Transition_VAE(128, 128*5, 8*5, 128, 5, 128)
# loss, loss_recon, kl_divergence = model.loss_function(s_1, o, a, r)
# print(loss, loss_recon, kl_divergence)