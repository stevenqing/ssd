
from torch import nn
import numpy as np
import gym
import torch
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out) 
    
class CBAM_Gaussian(nn.Module):
    def __init__(self, 
                 observation_space: gym.spaces.Box, 
                 channel=18, 
                 features_dim=128,
                 view_len=7, 
                 num_frames=6, 
                 fcnet_hiddens=[1024, 128], 
                 reduction=16, 
                 kernel_size=7):
        
        super(CBAM_Gaussian, self).__init__(observation_space,features_dim)

        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

        flat_out = num_frames * 3 * (view_len * 2 + 1) ** 2 # eliminate the padding?
        self.fc_mu = nn.Linear(flat_out, features_dim)
        self.fc_logsigma = nn.Linear(flat_out, features_dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.ca(x)
        x = self.sa(x)

        # flatten features
        features = torch.flatten(F.relu(x), start_dim=1)
        mu = self.fc_mu(features)
        logsigma = self.fc_logsigma(features)
        return mu, logsigma
    

class Inverse_ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class Inverse_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.ConvTranspose2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out) 


class Inverse_CBAM_Gaussian(nn.Module):
    def __init__(self, 
                 action_space: gym.spaces.Discrete, 
                 features_dim=128, 
                 view_len=7, 
                 num_frames=6, 
                 channel=18,
                 fcnet_hiddens=[1024, 128], 
                 reduction=16, 
                 kernel_size=7):
        
        super(Inverse_CBAM_Gaussian, self).__init__(action_space,features_dim)

        self.inverse_ca = Inverse_ChannelAttention(channel, reduction)
        self.inverse_sa = Inverse_SpatialAttention(kernel_size)

        flat_out = num_frames * 3 * (view_len * 2 + 1) ** 2
        self.fc = nn.Linear(features_dim, flat_out)