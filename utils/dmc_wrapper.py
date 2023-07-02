"""Code adjusted from: https://github.com/imgeorgiev/dmc2gymnasium
"""
import logging
import os

import numpy as np
from dm_env import specs
from dm_control import suite
from gymnasium.core import Env
from gymnasium.spaces import Box


def _spec_to_box(spec, dtype=np.float32):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        else:
            logging.error("Unrecognized type")

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return Box(low, high, dtype=dtype)


def _flatten_obs(obs, dtype=np.float32):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0).astype(dtype)


class EnvSpec:
    def __init__(self, id):
        # To capture custom spec id names
        self.id = id


class DMControlWrapper(Env):
    def __init__(
        self,
        domain,
        task,
        task_kwargs={},
        environment_kwargs={},
        render_kwargs={},
        rendering="egl",
    ):
        """TODO comment up"""

        # for details see https://github.com/deepmind/dm_control
        assert rendering in ["glfw", "egl", "osmesa"]
        os.environ["MUJOCO_GL"] = rendering

        self._env = suite.load(
            domain,
            task,
            task_kwargs,
            environment_kwargs,
        )

        # placeholder to allow built in gymnasium rendering
        self.render_mode = "rgb_array"

        self._observation_space = _spec_to_box(self._env.observation_spec().values())
        self._action_space = _spec_to_box([self._env.action_spec()])
        self._spec = EnvSpec(f"{domain}_{task}")

        # set seed if provided with task_kwargs
        if "random" in task_kwargs:
            seed = task_kwargs["random"]
            self._observation_space.seed(seed)
            self._action_space.seed(seed)

        # Set camera height and width if provided with render_kwargs
        if "height" in render_kwargs and "width" in render_kwargs:
            self.render_height = render_kwargs["height"]
            self.render_width = render_kwargs["width"]

        if "camera_id" in render_kwargs:
            self.render_camera_id = render_kwargs["camera_id"]
        else:
            self.render_camera_id = 0

    def __getattr__(self, name):
        """Add this here so that we can easily access attributes of the underlying env"""
        return getattr(self._env, name)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def spec(self):
        return self._spec

    @property
    def reward_range(self):
        """Always has a per-step reward range of (0, 1)"""
        return 0, 1

    def step(self, action):
        if action.dtype.kind == "f":
            action = action.astype(np.float32)
        assert self._action_space.contains(action)

        timestep = self._env.step(action)
        observation = _flatten_obs(timestep.observation)
        reward = timestep.reward
        termination = False  # we never reach a goal
        truncation = timestep.last()
        info = {"discount": timestep.discount}
        return observation, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        if seed:
            logging.warn(
                "Currently there is no way of seeding episodes. It only allows to seed experiments on environment initialization"
            )

        if options:
            logging.warn("Currently doing nothing with options={:}".format(options))
        timestep = self._env.reset()
        observation = _flatten_obs(timestep.observation)
        info = {}
        return observation, info

    def render(self):
        return self._env.physics.render(
            height=self.render_height,
            width=self.render_width,
            camera_id=self.render_camera_id,
        )
