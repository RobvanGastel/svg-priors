import yaml
import argparse

import torch
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from utils.misc import make_gif
from utils.logger import Logger
from algos.svg0.buffer import RolloutBuffer
from algos.svg0.agent import SVG0


def main(config):
    Logger.get().info(f"Start training, experiment name: {config['name']}")
    Logger.get().info(f"config: {config}")

    env = RecordEpisodeStatistics(gym.make(config["env"]))
    test_env = RecordEpisodeStatistics(gym.make(config["env"], render_mode="rgb_array"))

    agent = SVG0(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_lim=env.action_space.high[0],
        **config["svg0"],
    ).to(config["device"])

    buffer = RolloutBuffer(
        config["svg0"]["buffer_steps"],
        env.observation_space.shape[0],
        env.action_space.shape[0],
        config["device"],
    )

    global_step = 0
    for episode in range(config["epochs"]):
        obs, _ = env.reset()
        termination, truncated = False, False

        while not (termination or truncated):
            obs = torch.tensor(obs).to(config["device"])
            act = agent.act(obs)
            next_obs, rew, termination, truncated, info = env.step(act.cpu().numpy())

            buffer.store(obs, act, rew, next_obs, termination)

            obs = next_obs

            if termination or truncated:
                if episode % 5 == 0 and episode != 0:
                    batch = buffer.get()
                    agent.optimize(batch, global_step)
                    buffer.reset()

                Logger.get().info(
                    f"episode #: {episode} "
                    f"time elapsed: {np.mean(info['episode']['t']):.1f} "
                    f"episode return: {np.mean(info['episode']['r']):.3f} "
                    f" and episode length: {np.mean(info['episode']['l']):.0f}"
                )
                break
            global_step += 1

        # Log final episode reward
        Logger.get().writer.add_scalar("env/return", info["episode"]["r"], global_step)
        Logger.get().writer.add_scalar("env/length", info["episode"]["l"], global_step)

        # Store the weights of the agent and make a gif
        if episode % config["log_every_n"] == 0 and episode != 0:
            make_gif(agent, test_env, episode, config)
        #     agent.save_weights(config["path"], epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str)
    parser.add_argument("-c", "--config", type=str, default="configs/svg0.yml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert args.name is not None, "Pass a name for the experiment"
    config["name"] = args.name

    # CUDA device
    config["device"] = torch.device(config["device_id"])

    # Initialize logger
    config["path"] = f"runs/{args.name}"
    Logger(args.name, config["path"])

    main(config)
