import gym
import copy
import itertools
import collections

import torch
import torch.nn as nn
from torch.optim import Adam

from buffer import RolloutBuffer
from core import MLPVFunction, MLPQFunction, MLPGaussianActor


VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages q_estimate")

class SVG0:
    def __init__(self, config, env, gamma=0.99, polyak=0.995):

        # TODO: Discount factor, buffer size, hidden_sizes
        hidden_sizes=(64, 64)
        activation=nn.Tanh
        
        self.gamma = gamma
        self.polyak = polyak
        self.update_step = 0
        self.update_target_every = 50
        
        self.env = env
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        print(obs_dim, act_dim)
        
        # Replay buffer
        self.buffer = RolloutBuffer(obs_dim, act_dim, 200)
        
        
        # Policy Network
        self.pi = MLPGaussianActor(
            obs_dim, act_dim, hidden_sizes, activation
        )

        # Value functions
        self.critic = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.baseline = MLPVFunction(obs_dim, hidden_sizes, activation)
        self.target_critic = MLPVFunction(obs_dim, hidden_sizes, activation)
        

        # Set up optimizers for policy and value function
        self.actor_optim = Adam(self.pi.parameters(), lr=5e-4)
        
        value_params = itertools.chain(self.critic.parameters(), self.baseline.parameters())
        self.critic_optim = Adam(value_params, lr=5e-4)
        self.value_params = value_params

    def update_target(self):
        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.baseline.parameters(), self.target_critic.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
                
                
    def act(self, obs):
        pi = self.pi._distribution(obs)
        a = pi.sample()
        return a
    
    def update(self):
        data = self.buffer.get()
        
        self.update_step += 1
        
        for _ in range(3):
            self.actor_optim.zero_grad()
            loss_actor, actor_info = self.update_actor(data)
            loss_actor.backward()
        
            nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
                
            self.actor_optim.step()
            
            self.critic_optim.zero_grad()
            loss_critic, critic_info = self.update_critic(data)
            loss_critic.backward()
            
            nn.utils.clip_grad_norm_(self.value_params, 1.0)
            
            self.critic_optim.step()
        
        if self.update_step % self.update_target_every == 0:
            self.update_target()
            
    
    def update_critic(self, data):
        
        obs, act, rew, next_obs = data['obs'], data['act'], data['rew'], data['next_obs']
        done = data['done']
        
        with torch.no_grad():
            v_t = self.baseline(next_obs)
        v_tm1 = self.baseline(obs).squeeze()
        q_tm1 = self.critic(obs, act).squeeze()
        discount_t = (1 - done) * self.gamma

        vtrace_target = self.vtrace_td_errors_and_advantage(v_tm1.detach(), 
                                    discount_t, rew, v_t, q_tm1)
        
        q_loss = (vtrace_target.q_estimate - q_tm1) * (1 - self.gamma)
        # + utils.l2_norm(
            # self.model.critic.named_parameters())
        td_loss = (vtrace_target.vs - v_tm1) * (1 - self.gamma)
        
        
        loss = 0.5 * q_loss.pow(2) + 0.5 * td_loss.pow(2)
        
        return loss.mean(), {"critic/td": td_loss.mean().detach(),
                             "critic/q_loss": q_loss.mean().detach()
                             }


    def vtrace_td_errors_and_advantage(self, log_rhos, discounts, rewards, 
                                    values, bootstrap_value, 
                                    clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
        
        with torch.no_grad():
            rhos = torch.exp(log_rhos)
            if clip_rho_threshold is not None:
                clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
            else:
                clipped_rhos = rhos

            cs = torch.clamp(rhos, max=1.0)
            # print(bootstrap_value.shape)
            # print(values[1:].shape)
            
            # print(torch.cat((values[1:], bootstrap_value[-1])).shape)
            # Append bootstrapped value to get [v1, ..., v_t+1]
            # values_t_plus_1 = torch.cat(
            #     [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
            # )
            values_t_plus_1 = values
            # torch.cat((values[1:], bootstrap_value[-1]), dim=0)
            deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

            acc = torch.zeros_like(bootstrap_value)
            result = []
            for t in range(discounts.shape[0] - 1, -1, -1):
                acc = deltas[t] + discounts[t] * cs[t] * acc
                result.append(acc)
            result.reverse()
            vs_minus_v_xs = torch.stack(result)

            # Add V(x_s) to get v_s.
            vs = torch.add(vs_minus_v_xs, values)

            # Advantage for policy gradient.
            broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
            
            vs_t_plus_1 = torch.cat(
                [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
            )
            
            
            q_estimate = rewards + discounts * vs_t_plus_1
            if clip_pg_rho_threshold is not None:
                clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
            else:
                clipped_pg_rhos = rhos
                
            pg_advantages = clipped_pg_rhos * (q_estimate - values)

            # Make sure no gradients backpropagated through the returned values.
            return VTraceReturns(vs=vs, pg_advantages=pg_advantages, q_estimate=q_estimate)
    
        
    def update_actor(self, data):

        obs, act, rew, next_obs = data['obs'], data['act'], data['rew'], data['next_obs']
        done = data['done']
        
        pi = self.pi(obs)[0]
        new_actions = pi.rsample()

        # reg = utils.l2_norm(self.model.actor.named_parameters())
        value = (1. - done) * self.critic(obs, new_actions).squeeze(-1)
        return (-value).mean(), {"actor/value": value.mean().detach(),
                                 "actor/loc": pi.mean.mean(),
                                 "actor/scale": pi.variance.mean()}


    def train_agent(self, epochs=3000):
        print("Train agent")

        # TODO: Continuous environment
        env = gym.make("Pendulum-v0")
        act_dim = env.action_space.shape
        obs_dim = env.observation_space.shape
        
        global_step = 0

        # 1) Given empty experience database D
        for episode in range(epochs):
            obs, ep_ret, ep_len = self.env.reset(), 0, 0

            # 2) for t = 0 to ∞ do
            while True:
                # 3) Apply control π(s,η; θ), η ∼ ρ(η)
                act = self.act(torch.tensor(obs).float())
                
                # 4) Observe r, s′
                next_obs, rew, done, info = self.env.step(act)
                
                # d = False if ep_len==max_ep_len else d
                
                # 5) Insert (s, a, r, s′) into D
                self.buffer.store(obs, act, rew, next_obs, done)

                # Model and critic updates 
                # 7) Train generative model f ˆ using D
                
                
                # 8) Train value function ˆ V using D (Alg. 4)

                ep_len += 1
                ep_ret += rew
                
                obs = next_obs
                
                # 9) Policy update
                
                # if global_step % 1990 == 0:
                      
                # 10) Sample (s_k, a_k, r_k, s_k+1) from D (k ≤ t)
                # w = p(a_k |s_k; θ_t) / p(a_k |s_k; θ_k)               
                if done:
                    print(f"Episode length: {ep_len} and return: {ep_ret}")
                    print("Episode finished after %i steps" % ep_len)
                    
                    self.update()
                    break
                
                global_step += 1

print("Init SVG0")
env = gym.make("Pendulum-v0")
svg = SVG0(None, env)
print("Init SVG0")

svg.train_agent()