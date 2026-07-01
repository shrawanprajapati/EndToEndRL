"""
models/noisy_net.py — Shared NoisyNet module
Factorized Gaussian NoisyLinear layers for exploration in PPO.
Used by both train_ppo.py and train_final.py.
"""

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, std_init=0.5):
        super(NoisyLinear, self).__init__(in_features, out_features, bias)
        self.std_init = std_init
        self.weight_mu = nn.Parameter(self.weight.data.clone())
        self.weight_sigma = nn.Parameter(th.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', th.empty(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(self.bias.data.clone())
            self.bias_sigma = nn.Parameter(th.empty(out_features))
            self.register_buffer('bias_epsilon', th.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
            self.register_buffer('bias_epsilon', None)
        self.reset_noisy_parameters()
        self.reset_noise()

    def reset_parameters(self):
        if hasattr(self, 'weight'):
            super(NoisyLinear, self).reset_parameters()

    def reset_noisy_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = th.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, input):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon if self.bias_mu is not None else None
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


def replace_linear_with_noisy_and_norm(module):
    """Recursively replace all nn.Linear layers with a sequence of NoisyLinear and LayerNorm."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, NoisyLinear):
            noisy = NoisyLinear(child.in_features, child.out_features, bias=child.bias is not None)
            norm = nn.LayerNorm(child.out_features)
            seq = nn.Sequential(noisy, norm)
            setattr(module, name, seq)
        else:
            replace_linear_with_noisy_and_norm(child)


def replace_linear_with_noisy(module):
    """Recursively replace all nn.Linear layers with NoisyLinear."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, NoisyLinear):
            noisy = NoisyLinear(child.in_features, child.out_features, bias=child.bias is not None)
            setattr(module, name, noisy)
        else:
            replace_linear_with_noisy(child)


class NoisyActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with hidden layers replaced by NoisyLinear + LayerNorm, and output layers by NoisyLinear."""
    def __init__(self, *args, **kwargs):
        super(NoisyActorCriticPolicy, self).__init__(*args, **kwargs)
        replace_linear_with_noisy_and_norm(self.mlp_extractor)
        replace_linear_with_noisy(self.action_net)
        replace_linear_with_noisy(self.value_net)
