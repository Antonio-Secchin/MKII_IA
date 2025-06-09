import sys
import retro
from retro import Observations

import gymnasium as gym

import numpy as np

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

from gymnasium import Wrapper

from model_callback import EvalCallback

class LastActionsWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, n_actions):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)
        self.n_actions = n_actions

        sample = self.env.action_space.sample()
        self.action_size = sample.size if isinstance(sample, np.ndarray) else 1

        self.last_actions = np.zeros((self.n_actions, self.action_size))
        self.num_actions = 0

        orig_obs_space = self.env.observation_space
        obs_size = orig_obs_space.shape[0] if isinstance(orig_obs_space, gym.spaces.Box) else len(orig_obs_space)
        new_obs_dim = obs_size + self.n_actions * self.action_size

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32
        )  

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        obs, info = self.env.reset(**kwargs)
        self.last_actions = np.zeros((self.n_actions,self.action_size))
        obs = np.concatenate([obs, self.last_actions.flatten()])
        self.num_actions = 0

        return obs, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is this a final state (episode finished),
        is the max number of steps reached (episode finished artificially), additional informations
        """
        if isinstance(action, (int, np.integer)):
            # convert to 1D array if needed
            action_vec = np.array([action])
        else:
            action_vec = action


        self.last_actions[self.num_actions%self.n_actions] = action_vec
        self.num_actions += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.concatenate([obs, self.last_actions.flatten()])

        return obs, reward, terminated, truncated, info