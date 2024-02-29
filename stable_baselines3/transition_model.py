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
class P_s_za(nn.Module):
    def __init__(self, dim_z=128, dim_a=8, nh=2, dim_h=128, dim_out=128):
        super().__init__()
        self.nh = nh
        self.dim_out = dim_out

        # Hidden layers
        self.hidden_z = nn.Sequential(nn.Linear(dim_z, dim_h), nn.ReLU())
        self.hidden_a = nn.Sequential(nn.Linear(dim_a, dim_h), nn.ReLU())
        self.hidden_layer = nn.Sequential(nn.Linear(dim_h*2, dim_h), nn.ReLU())

        self.output = nn.ModuleList(
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

    def forward(self, z, a):
        z = self.hidden_z(z).squeeze(1)
        a = self.hidden_a(a)
        x = torch.cat([z, a], 1)
        x = self.hidden_layer(x)
        return x





####### Encoder #######
# the input of the encoder should be the state, observation, action, reward
# the output of the encoder should be the latent variable z

class Q_z_xot(nn.Module):
    def __init__(self, dim_s=128, dim_in_oa=128*5+8*5, nh=2, dim_h=128, dim_out=128):
        super().__init__()
        # dim in is dim of x + dim of y
        # dim_out is dim of latent space z
        # save required vars
        self.nh = nh
        self.dim_out = dim_out
        # Shared layers with separated output layers
        self.hidden_state = nn.Sequential(nn.Linear(dim_s, dim_h), nn.ReLU())
        self.hidden_oa = nn.Sequential(nn.Linear(dim_in_oa, dim_h), nn.ReLU())

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

    def forward(self, s, oa) -> Normal:
        s = self.hidden_state(s)
        oa = self.hidden_oa(oa)
        x = torch.cat([s, oa], 1)
        x = self.hidden_layer(x)

        mu = torch.stack([mu(x) for mu in self.mu_header]).swapaxes(0, 1)
        sigma = torch.stack([sigma(x) for sigma in self.sigma_header]).swapaxes(0, 1)
        z = Normal(mu, torch.exp(sigma))
        return z


class Transition_VAE(nn.Module):
    def __init__(self, hidden_size, dim_o, dim_a, dim_s, dim_z, num_agents): # dim_s = dim_z
        super().__init__()
        nh=3
        self.encoder = Q_z_xot(dim_s=dim_s, dim_in_oa=dim_o+dim_a, nh=nh, dim_h=hidden_size, dim_out=dim_z)
        self.decoder = P_s_za(dim_z=dim_z,dim_a=dim_a,nh=nh, dim_h=hidden_size, dim_out=dim_z)

    def forward(self, s, o, a):
        oa = torch.cat([o, a], -1)
        z_dist = self.encoder(s, oa)
        z = z_dist.rsample()
        y_dist = self.decoder(z, a)
        return y_dist, z_dist


    def loss_function(self, s_1, s_2, o, a):
        # Compute z distribution
        oa = torch.cat([o, a], -1)
        z_dist = self.encoder(s_1, oa)
        z = z_dist.rsample()

        # Compute prediction distribution
        y_dist = self.decoder(z, a)

        # Calculate the loss
        loss_recon = F.mse_loss(y_dist, s_2, reduction='mean')

        # Compute total loss
        kl_divergence = 0.5 * (1 + 2 * z_dist.scale.log() - z_dist.mean.pow(2) - z_dist.scale.pow(2)).sum(-1).mean()
        loss = loss_recon + kl_divergence

        return -loss, loss_recon, kl_divergence


s_1 = torch.rand(32, 128)
s_2 = torch.rand(32, 128)
o = torch.rand(32, 128*5)
a = torch.rand(32, 8*5)
model = Transition_VAE(128, 128*5, 8*5, 128, 128, 5)
loss, loss_recon, kl_divergence = model.loss_function(s_1, s_2, o, a)
print(loss, loss_recon, kl_divergence)