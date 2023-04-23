import torch


class RolloutBuffer:
    def __init__(
        self,
        size,
        obs_dim,
        action_dim,
        device=None,
    ):
        """
        A buffer for storing trajectories experienced by an SVG-0 agent interacting
        with the environment.
        """

        self.data = {
            "obs": torch.zeros((size, obs_dim)).to(device),
            "next_obs": torch.zeros((size, obs_dim)).to(device),
            "action": torch.zeros((size, action_dim)).to(device),
            "reward": torch.zeros((size)).to(device),
            "termination": torch.zeros((size,)).to(device),
        }

        self.max_size, self.device, self.ptr = size, device, 0

    def store(self, obs, action, rew, next_obs, termination):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """

        curr_ptr = self.ptr % self.max_size
        self.data["obs"][curr_ptr] = obs
        self.data["action"][curr_ptr] = torch.tensor(action).to(self.device)
        self.data["reward"][curr_ptr] = torch.tensor(rew).to(self.device)
        self.data["next_obs"][curr_ptr] = torch.tensor(next_obs).to(self.device)
        self.data["termination"][curr_ptr] = torch.tensor(termination).to(self.device)

        self.ptr = (self.ptr + 1) % self.max_size

    def reset(self):
        self.ptr = 0
        for key, val in self.data.items():
            self.data[key] = torch.zeros_like(val)

    def get(self):
        """
        Obtain entire buffer to random sample from during the agent optimization.
        """
        return {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in self.data.items()
        }
