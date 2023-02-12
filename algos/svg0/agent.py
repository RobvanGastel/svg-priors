import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import Logger
from algos.svg0.core import StochasticPolicy, DoubleQFunction
from torch.utils.data import BatchSampler, SubsetRandomSampler


class SVG0(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_lim,
        seed=0,
        lr=3e-4,
        update_epochs=5,
        num_mini_batches=5,
        hidden_sizes=[64, 64],
        activation=nn.ReLU,
        update_target_every=2,
        gamma=0.95,
        polyak=0.95,
        **kwargs,
    ):
        super().__init__()
        torch.manual_seed(seed)

        # Optimize variables
        self.update_step = 0
        self.gamma = gamma
        self.polyak = polyak
        self.action_limit = action_lim
        self.update_epochs = update_epochs
        self.num_mini_batches = num_mini_batches
        self.update_target_every = update_target_every

        # Policy network
        self.pi = StochasticPolicy(
            obs_dim, action_dim, action_lim, hidden_sizes, activation
        )

        # Q functions networks
        self.q = DoubleQFunction(obs_dim, action_dim, hidden_sizes, activation)

        self.target_q = copy.deepcopy(self.q)
        self.target_q.eval()
        for p in self.target_q.parameters():
            p.requires_grad = False

        # Set up optimizers for policy and value function
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.policy_optim = torch.optim.Adam(self.pi.parameters(), lr=lr)

    def update_target(self):
        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.q.parameters(), self.target_q.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def act(self, obs, deterministic=False):

        with torch.no_grad():
            action, _, _ = self.pi(obs, deterministic=deterministic)

        return action

    def optimize(self, batch, global_step):

        for _ in range(self.update_epochs):
            self.update_step += 1

            batch_indices = BatchSampler(
                SubsetRandomSampler(range(batch["obs"].size(0))),
                self.num_mini_batches,
                False,
            )
            for minibatch_indices in batch_indices:
                minibatch = {k: v[minibatch_indices] for k, v in batch.items()}

                # Update Q functions
                q_loss = self._compute_q_loss(minibatch)

                self.q_optim.zero_grad()
                # nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
                q_loss.backward()
                self.q_optim.step()

                # Update stochastic policy
                for p in self.q.parameters():
                    p.requires_grad = False

                pi_loss = self._compute_policy_loss(minibatch)
                self.policy_optim.zero_grad()
                pi_loss.backward()

                # nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
                self.policy_optim.step()
                for p in self.q.parameters():
                    p.requires_grad = True

        if self.update_step % self.update_target_every == 0:
            self.update_target()

        # Logging of the agent variables
        Logger.get().writer.add_scalar("SVG0/q_loss", q_loss.item(), global_step)
        Logger.get().writer.add_scalar("SVG0/policy_loss", pi_loss.item(), global_step)

    def _compute_q_loss(self, batch):

        b_obs, b_act, b_rew, b_next_obs, b_done = (
            batch["obs"],
            batch["action"],
            batch["reward"],
            batch["next_obs"],
            batch["termination"],
        )

        with torch.no_grad():
            b_next_action, _, b_next_mean = self.pi(b_next_obs, with_logp=False)

            target_noise = b_next_action - b_next_mean
            b_next_action = b_next_mean + target_noise

            q_t1, q_t2 = self.target_q(b_next_obs, b_next_action)

            # take min to mitigate maximization bias in q-functions
            q_target = torch.min(q_t1, q_t2)

            # TD target
            value_target = b_rew + (1.0 - b_done) * self.gamma * q_target

        q_1, q_2 = self.q(b_obs, b_act)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1 + loss_2

    def _compute_policy_loss(self, batch):
        b_obs = batch["obs"]

        b_action, _, _ = self.pi(b_obs, with_logp=False)
        q_b1, q_b2 = self.q(b_obs, b_action)
        b_q_values = torch.min(q_b1, q_b2)

        policy_loss = (-b_q_values).mean()
        return policy_loss
