import gym
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.optim import Adam

from core import MLPVFunction, MLPQFunction, MLPGaussianActor


class SVG0(nn.Module):
    def __init__(self, config, env):
        self.__init__()

        # TODO: Discount factor
        self.env = env
        
        # TODO: Target + online networks?
        # Policy network
        self.pi_network = MLPGaussianActor(
            config.obs_dim, config.act_dim, config.hidden_sizes
        )
        # Critic network
        self.critic_network = MLPVFunction(
            config.obs_dim, config.hidden_sizes, config.activation
        )
        # Prior network
        self.prior_network = MLPGaussianActor(
            config.obs_dim, config.act_dim, config.hidden_sizes
        )
        
        self.target_pi_network = copy.deepcopy(self.pi_network)
        self.target_critic_network = copy.deepcopy(self.critic_network)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.pi_network.parameters(), lr=config.lr)
        self.prior_optimizer = Adam(self.prior_network.parameters(), lr=config.lr)
        self.critic_optimizer = Adam(self.critic_network.parameters(), lr=config.lr)

    def act(self, obs):
        return NotImplemented

    def update_step(self):
        return NotImplemented

    def train_agent(self, epochs=50):

        # TODO: Continuous environment
        env = gym.make("CartPole-v1")

        # 1) Given empty experience database D
        for episode in range(epochs):
            o, ep_ret, ep_len = self.env.reset(), 0, 0

            # 2) for t = 0 to ∞ do
            while True:
                # 3) Apply control π(s,η; θ), η ∼ ρ(η)
                action = self.act(obs)
                
                # 4) Observe r, s′
                obs, rew, done, info = self.env.step(action)
                
                # 5) Insert (s, a, r, s′) into D
                

                # Model and critic updates 
                # 7) Train generative model f ˆ using D 
                # 8) Train value function ˆ V using D (Alg. 4)

                ep_len += 1
                ep_ret += rew
                # 9) Policy update
                # 10) Sample (s_k, a_k, r_k, s_k+1) from D (k ≤ t)
                
                # w = p(a_k |s_k; θ_t) / p(a_k |s_k; θ_k)               
                if done:
                    print("Episode finished after %i steps" % ep_len)
                    break
       