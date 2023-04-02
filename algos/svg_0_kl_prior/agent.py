import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.utils.data import BatchSampler, RandomSampler

from utils.logger import Logger
from algos.svg_0_kl_prior.core import StochasticPolicy, DoubleQFunction


class SVG0(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_lim,
        lr=3e-4,
        tau=5e-3,
        gamma=0.95,
        batch_size=5,
        update_epochs=5,
        update_interval=2,
        activation=nn.ReLU,
        hidden_sizes=[64, 64],
        prior_obs_dim=1,
        prior_coeff=0.01,
        **kwargs,
    ):
        super().__init__()

        # Optimize variables
        self.update_step = 0
        self.gamma = gamma
        self.tau = tau
        self.prior_coeff = prior_coeff
        self.max_grad_norm = 0.5  # TODO: Does norm help?
        self.action_limit = action_lim
        self.update_epochs = update_epochs
        self.update_interval = update_interval
        self.batch_size = batch_size

        # Policy network
        self.pi = StochasticPolicy(
            obs_dim, action_dim, action_lim, hidden_sizes, activation
        )

        # Prior policy network
        self.prior_pi = StochasticPolicy(
            prior_obs_dim, action_dim, action_lim, hidden_sizes, activation
        )

        # Q functions networks
        self.q = DoubleQFunction(obs_dim, action_dim, hidden_sizes, activation)

        self.target_q = copy.deepcopy(self.q)
        self.target_q.eval()
        for p in self.target_q.parameters():
            p.requires_grad = False

        # Set up optimizers for policy and value function
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.prior_optim = torch.optim.Adam(self.prior_pi.parameters(), lr=lr)

        self.pi_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.pi_optim, 1, 1e-6, total_iters=3000
        )
        self.q_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.q_optim, 1, 1e-6, total_iters=3000
        )

    def update_target(self):
        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p_targ, p in zip(self.target_q.parameters(), self.q.parameters()):
                p_targ.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_targ.data)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            action, _, mean = self.pi(obs)

        if deterministic:
            action = mean

        return action

    def optimize(self, batch, global_step):
        for _ in range(self.update_epochs):
            self.update_step += 1

            batch_indices = BatchSampler(
                RandomSampler(range(batch["obs"].size(0))),
                self.batch_size,
                False,
            )

            batch_indices = next(iter(batch_indices))
            batch = {k: v[batch_indices] for k, v in batch.items()}

            # Compute KL divergence between pi and prior pi
            with torch.no_grad():
                _, _, _, pi = self.pi(batch["obs"])
            _, _, _, prior_pi = self.prior_pi(batch["obs"][:, :1])
            kl_reg = kl_divergence(pi, prior_pi)
            loss_kl = kl_reg.sum()

            # Update prior goal-agnostic policy
            self.prior_optim.zero_grad()
            loss_kl.backward()
            self.prior_optim.step()

            # Update Q functions
            q_loss, q_1, q_2 = self._compute_q_loss(batch, kl_reg)

            self.q_optim.zero_grad()
            nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
            q_loss.backward()
            self.q_optim.step()

            if self.update_step % self.update_interval == 0:

                for p in self.q.parameters():
                    p.requires_grad = False

                # Update stochastic policy
                pi_loss, pi = self._compute_policy_loss(batch)
                pi_loss = pi_loss - self.prior_coeff * loss_kl.detach()

                self.pi_optim.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), self.max_grad_norm)
                self.pi_optim.step()

                for p in self.q.parameters():
                    p.requires_grad = True

            self.update_target()

        self.q_scheduler.step()
        self.pi_scheduler.step()

        # Logging of the agent variables
        writer = Logger.get().writer
        writer.add_scalar("SVG0/Q_loss", q_loss.item(), global_step)
        writer.add_scalar("SVG0/policy_loss", pi_loss.item(), global_step)
        writer.add_histogram("SVG0/policy_histogram", pi, global_step)
        writer.add_histogram("SVG0/Q1_histogram", q_1, global_step)
        writer.add_histogram("SVG0/Q2_histogram", q_2, global_step)

    def _compute_q_loss(self, batch, kl_reg):

        b_obs, b_act, b_rew, b_next_obs, b_done = (
            batch["obs"],
            batch["action"],
            batch["reward"],
            batch["next_obs"],
            batch["termination"],
        )

        with torch.no_grad():
            b_next_action, next_log_probs, b_next_mean, _ = self.pi(
                b_next_obs, with_logp=False
            )

            target_noise = b_next_action - b_next_mean
            b_next_action = b_next_mean + target_noise

            q_t1, q_t2 = self.target_q(b_next_obs, b_next_action)

            # take min to mitigate maximization bias in q-functions
            q_target = torch.min(q_t1, q_t2)

            # TD target
            value_target = (b_rew - self.prior_coeff * kl_reg) + (
                1.0 - b_done
            ) * self.gamma * q_target

        q_1, q_2 = self.q(b_obs, b_act)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1 + loss_2, q_1, q_2

    def _compute_policy_loss(self, batch):
        b_obs = batch["obs"]

        b_action, _, _, _ = self.pi(b_obs, with_logp=True)
        q_b1, q_b2 = self.q(b_obs, b_action)
        b_q_values = torch.min(q_b1, q_b2)

        policy_loss = (-b_q_values).mean()
        return policy_loss, b_action