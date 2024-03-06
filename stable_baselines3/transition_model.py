import argparse
import copy
import random
import time
from stable_baselines3.CBAM import CBAM_Gaussian
from stable_baselines3.CNN import CNN_Encoder, CNN_Decoder
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
        
    def forward(self, z, a) -> Normal:
        # Output layers
        return recon



class Q_z_oar(nn.Module):
    def __init__(self, dim_o=128*5+8*5, dim_r=5, nh=2, dim_h=128, dim_out=128):
        super().__init__()
        self.nh = nh
        self.dim_out = dim_out
        # Shared layers with separated output layers

        self.cnn_encoder = CBAM_Gaussian(dim_o, channel=18, features_dim=128, view_len=7, num_frames=6, fcnet_hiddens=[1024, 128], reduction=16, kernel_size=7)


    def forward(self, o, a, r) -> Normal:
        mu, sigma = self.cbam_gaussian(o)
        return mu, sigma
    
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