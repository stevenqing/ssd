from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class P_r_xzt(nn.Module):
    def __init__(self, dim_in=20, nh=3, dim_h=20, dim_out=1, nA=2):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # Separated forwards for different t values, TAR
        self.hidden = nn.Sequential(nn.Linear(dim_in, dim_h), nn.ReLU())
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.mu_t = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())
                        for _ in range(nh)
                    ],
                    nn.Linear(dim_h, dim_out),
                )
                for _ in range(nA)
            ]
        )
        self.sigma_t = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())
                        for _ in range(nh)
                    ],
                    nn.Linear(dim_h, dim_out),
                )
                for _ in range(nA)
            ]
        )

    def forward(self, xz, t):
        # Separated forwards for different t values, TAR
        x = self.hidden(xz)
        t = t.type(torch.int64)
        mu = torch.stack([mu(x) for mu in self.mu_t]).swapaxes(0, 1)
        t = t.unsqueeze(2).expand(-1, -1, mu.shape[-1]).view(-1,1,4)
        t_transform = torch.tensor([[1,1,0,0],[1,0,0,1],[0,1,1,0],[0,0,1,1]]).expand(t.size(0),4,4)
        t = torch.bmm(t,t_transform).view(-1,4,1)
        mu = mu*t
        sigma = torch.stack([sigma(x) for sigma in self.sigma_t]).swapaxes(0, 1)
        sigma = sigma*t
        return Normal(mu.sum(1).squeeze(-1),sigma.sum(1).exp().squeeze(-1))




####### Inference model / Encoder #######


class Q_z_xy(nn.Module):
    def __init__(self, dim_in=25 + 1, nh=3, dim_h=20, dim_out=20, nA=2):
        super().__init__()
        # dim in is dim of x + dim of y
        # dim_out is dim of latent space z
        # save required vars
        self.nh = nh
        self.dim_out = dim_out
        # Shared layers with separated output layers
        self.hidden = nn.Sequential(nn.Linear(dim_in, dim_h), nn.ReLU())
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.mu_header = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())
                        for _ in range(nh)
                    ],
                    nn.Linear(dim_h, dim_out),
                )
                for _ in range(nA)
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
                for _ in range(nA)
            ]
        )

    def forward(self, xy, t) -> Normal:
        # Shared layers with separated output layers
        # print('before first linear z_infer')
        # print(xy)
        t = t.type(torch.int64)
        x = self.hidden(xy)
        mu = torch.stack([mu(x) for mu in self.mu_header]).swapaxes(0, 1)
        t = t.unsqueeze(2).expand(-1, -1, mu.shape[-1])
        mu = mu*t
        sigma = torch.stack([sigma(x) for sigma in self.sigma_header]).swapaxes(0, 1)
        sigma = sigma*t
        z = Normal(mu.sum(1), torch.exp(sigma.sum(1)))
        return z


class ReturnVAE(nn.Module):
    def __init__(self, nA, nS):
        super().__init__()
        hidden_size = 64
        dim_h = 1
        nh=3
        self.q_z_xyrt = Q_z_xyt(dim_in=2 * nS+1,dim_out=dim_h,nh=nh, dim_h=hidden_size, nA=nA)
        self.p_y_xzt = P_y_xzt(dim_in=nS + dim_h,dim_h=128,nh=nh, dim_out=nS, nA=nA)
        self.p_r_xzt = P_r_xzt(dim_in=nS + dim_h,dim_h=hidden_size,nh=3, dim_out=1, nA=nA)
        self.p_t_xz = P_t_xz(dim_in=nS + dim_h, nh=nh,  nA=nA)

    def loss_function(self, x, y, r):
        # Compute z distribution
        z_cat = torch.cat((x, y, r.unsqueeze(-1)), 1)
        # transform the one-hot vector to index

        y_index = torch.argmax(y,-1)
        
        z_dist = self.q_z_xyrt(z_cat, t)
        z = z_dist.rsample()

        # Compute prediction distribution
        y_logits = self.p_y_xzt(torch.cat((x, z), 1), t)
        r_logits = self.p_r_xzt(torch.cat((x, z), 1), t)

        # Calculate the loss
        y_loss = Categorical(logits=y_logits).log_prob(y.argmax(1))

        r_loss = r_logits.log_prob(r).squeeze()
        loss1 = (y_loss+r_loss).mean()

        # Compute total loss
        kl_divergence = 0.5 * (1 + 2 * z_dist.scale.log() - z_dist.mean.pow(2) - z_dist.scale.pow(2)).sum(-1).mean()
        loss = loss1 + kl_divergence

        return -loss, y_loss.mean(), kl_divergence, r_loss.mean()