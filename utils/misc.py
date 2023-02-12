import os

import torch
import numpy as np
from utils.logger import Logger
from moviepy.editor import ImageSequenceClip


def make_gif(agent, env, episode, config):
    """Save gif of the model on a test environment"""

    obs, _ = env.reset()
    terminated, truncated = False, False
    steps = []
    rewards = []
    while not (terminated or truncated):
        steps.append(env.render())

        obs = torch.tensor(obs).float().to(config["device"])
        action = agent.act(obs, deterministic=False)
        action = np.clip(
            action.cpu().numpy(), env.action_space.low[0], env.action_space.high[0]
        )
        print(f"debug: action {action}")
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

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
