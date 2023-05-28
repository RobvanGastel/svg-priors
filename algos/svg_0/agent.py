import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import BatchSampler, RandomSampler

from utils.logger import Logger
from utils.misc import save_state
from algos.svg_0.core import StochasticPolicy, DoubleQFunction


class SVG0(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_lim,
        action_space,
        device,
        lr=3e-4,
        tau=5e-3,
        gamma=0.95,
        batch_size=256,
        update_epochs=5,
        update_interval=2,
        activation=nn.ReLU,
        hidden_sizes=[64, 64],
        target_entropy=None,
        use_target_entropy=False,
        **kwargs,
    ):
        super().__init__()

        # Optimize variables
        self.update_step = 0
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_limit = action_lim
        self.update_epochs = update_epochs
        self.update_interval = update_interval
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.use_target_entropy = use_target_entropy

        # Set clamp values
        self.action_low = torch.tensor(action_space.low).to(device)
        self.action_high = torch.tensor(action_space.high).to(device)

        # Policy network
        self.pi = StochasticPolicy(obs_dim, action_dim, hidden_sizes, activation)

        # Q functions networks
        self.q = DoubleQFunction(obs_dim, action_dim, hidden_sizes, activation)

        self.target_q = copy.deepcopy(self.q)
        self.target_q.eval()
        for p in self.target_q.parameters():
            p.requires_grad = False

        # Set up optimizers for policy and value function
        self.q_optim = optim.Adam(self.q.parameters(), lr=lr)
        self.pi_optim = optim.Adam(self.pi.parameters(), lr=lr)

        self.pi_scheduler = optim.lr_scheduler.LinearLR(
            self.pi_optim, 1, 1e-6, total_iters=3000
        )
        self.q_scheduler = optim.lr_scheduler.LinearLR(
            self.q_optim, 1, 1e-6, total_iters=3000
        )

        # Temperature for target entropy to compare with KL priors
        if self.use_target_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.temp_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def save_weights(self, path, episode):
        save_state(
            {
                "q": self.q.state_dict(),
                "pi": self.pi.state_dict(),
                "q_optim": self.q_optim.state_dict(),
                "pi_optim": self.pi_optim.state_dict(),
            },
            path,
            episode,
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

        action.clamp_(self.action_low, self.action_high)
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
            mini_batch = {k: v[batch_indices] for k, v in batch.items()}

            # Update Q functions
            q_loss, q_1, q_2 = self._compute_q_loss(mini_batch)

            self.q_optim.zero_grad()
            q_loss.backward()
            self.q_optim.step()

            if self.update_step % self.update_interval == 0:
                for p in self.q.parameters():
                    p.requires_grad = False

                # Update stochastic policy
                pi_loss, pi, temp_loss = self._compute_policy_loss(mini_batch)

                if self.use_target_entropy:
                    self.temp_optimizer.zero_grad()
                    temp_loss.backward()
                    self.temp_optimizer.step()

                self.pi_optim.zero_grad()
                pi_loss.backward()
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
        return loss_1 + loss_2, q_1, q_2

    def _compute_policy_loss(self, batch):
        b_obs = batch["obs"]

        b_action, b_logprobs, _ = self.pi(b_obs, with_logp=True)
        q_b1, q_b2 = self.q(b_obs, b_action)
        b_q_values = torch.min(q_b1, q_b2)

        if self.use_target_entropy:
            policy_loss = (self.log_alpha.exp() * b_logprobs - b_q_values).mean()
            temp_loss = (
                -self.log_alpha.exp()
                * (b_logprobs.detach() + self.target_entropy).mean()
            )
        else:
            policy_loss = (-b_q_values).mean()
            temp_loss = None

        return policy_loss, b_action, temp_loss
