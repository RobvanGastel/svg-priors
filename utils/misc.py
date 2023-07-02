import os
import math

import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.distributions import constraints
from torch.distributions.transforms import Transform

import numpy as np
from moviepy.editor import ImageSequenceClip

from utils.logger import Logger


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is numerically unstable
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(), AffineTransform(loc=-1, scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`. It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead. Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
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


def make_gif(agent, env, episode, config):
    """
    Generate a GIF of the agent's performance during an episode in a given environment.

    Args:
        agent (object): An instance of the agent whose performance will be recorded.
        env (object): An instance of the OpenAI gym environment.
        episode (int): The episode number to save recording.
        config (dict): A dictionary of configuration parameters.

    Returns:
        None.
    """

    obs, _ = env.reset()
    terminated, truncated = False, False
    steps = []
    rewards = []
    while not (terminated or truncated):
        obs = torch.tensor(obs).float().to(config["device"])

        action = agent.act(obs, deterministic=True)
        action = np.clip(
            action.cpu().numpy(), env.action_space.low, env.action_space.high
        )

        obs, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)
        steps.append(env.render())

    clip = ImageSequenceClip(steps, fps=30)
    save_dir = os.path.join(config["path"], "gifs")
    gif_name = f"{save_dir}/{env.spec.id}_epoch_{str(episode)}.gif"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    clip.write_gif(
        gif_name,
        fps=30,
        verbose=False,
        logger=None,
    )

    Logger.get().info(f"Generating GIF {gif_name}")


def save_state(state_dict, path, epoch=None, job_id=None):
    """Save the model and optimizer states using PyTorch"""

    model_file = os.path.join(path, f"e{epoch}_state") if epoch is not None else path

    # save the model (to temporary path if job_id is specified then rename)
    model_file_tmp = model_file if job_id is None else model_file + f"_{job_id}"
    torch.save(state_dict, model_file_tmp)
    if model_file_tmp != model_file:
        os.rename(model_file_tmp, model_file)
