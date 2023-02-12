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

        # TODO: On-policy or ring buffer?

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

        # assert self.ptr < self.max_size
        self.data["obs"][self.ptr] = obs
        self.data["action"][self.ptr] = action
        self.data["reward"][self.ptr] = torch.tensor(rew).to(self.device)
        self.data["next_obs"][self.ptr] = torch.tensor(next_obs).to(self.device)
        self.data["termination"][self.ptr] = torch.tensor(termination).to(self.device)
        self.ptr += 1

    def reset(self):
        self.ptr = 0
        for key, val in self.data.items():
            self.data[key] = torch.zeros_like(val)

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # buffer has to be full before you can get
        # assert self.ptr == self.max_size

        return {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in self.data.items()
        }
