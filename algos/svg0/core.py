import torch
import torch.nn as nn

import math
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus
from torch.distributions import Normal, TransformedDistribution


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class StochasticPolicy(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, obs_dim, action_dim, action_lim, hidden_sizes, activation):
        super().__init__()
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs, deterministic=False, with_logp=True):
        x = self.net(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)

        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)

        if deterministic:
            # Only used for evaluating policy at test time.
            action = mu
        else:
            # TODO: Apply before rsample?
            transforms = [TanhTransform(cache_size=1)]
            pi_distribution = TransformedDistribution(pi_distribution, transforms)

            # for reparameterization trick (mean + std * N(0,1))
            action = pi_distribution.rsample()

        if with_logp:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            logp_a = pi_distribution.log_prob(action).sum(axis=-1)
            # logp_a -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
        else:
            logp_a = None

        action = torch.tanh(action)
        action = self.action_lim * action

        return action, logp_a, action


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


# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is numerically unstable
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(), AffineTransform(loc=-1, scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2.0 * (math.log(2.0) - x - softplus(-2.0 * x))
