import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution

from utils.misc import mlp, TanhTransform


class StochasticPolicy(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, obs_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.action_dim = action_dim

        self.net = mlp(
            [obs_dim] + list(hidden_sizes) + [action_dim * 2],
            activation,
            output_activation=nn.Tanh,
        )

    def forward(self, obs, with_logp=True):

        mu_logstd = self.net(obs)
        if len(mu_logstd.size()) == 1:
            mu, log_std = mu_logstd.chunk(2)
        else:
            mu, log_std = mu_logstd.chunk(2, dim=1)

        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        pi_distribution = TransformedDistribution(pi_distribution, transforms)
        action = pi_distribution.rsample()

        if with_logp:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            logp_a = pi_distribution.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logp_a = None

        mean = torch.tanh(mu)

        return action, logp_a, mean, pi_distribution


class DoubleQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q_1 = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.q_2 = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        x_1 = self.q_1(x).squeeze(-1)
        x_2 = self.q_2(x).squeeze(-1)
        return x_1, x_2
