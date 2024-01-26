import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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
            # nn.Tanh() # TODO double check the activation function
        ) 
        self.input_dim = input_dim
        self.num_agent = num_agents
        if enable_causality:
            self.dynamic_mask = dynamic_mask
            if self.dynamic_mask:
                # input state, action
                self.mask_predictor = nn.Sequential(
                    nn.Linear(input_dim, 512),  # Fully connected layer
                    nn.ReLU(),
                    nn.Linear(512, 128),  # Fully connected layer
                    nn.ReLU(),
                    nn.Linear(128, input_dim * self.num_agents),  # Output layer
                    nn.Sigmoid(),
                    MaskActivation()
                )
            
            else:
                self.causal_mask = \
                    nn.Parameter(torch.ones(self.num_agent, input_dim), requires_grad=True)
                self.sh = 0.1

    def forward(self, x, test=False):
        # x: [batch_size, input_dim]
        if self.enable_causality:
            # x: [batch_size, 1, input_dim]
            # causal_mask: [1, num_agent, input_dim]
            # masked_input: [batch_size, num_agent, input_dim]
            if self.dynamic_mask:
                mask = self.mask_predictor(x).view(-1, self.num_agents, self.input_dim)
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