import sys
import retro
from retro import Observations

import gymnasium as gym

import numpy as np

from gymnasium import Wrapper


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

        self.last_actions = np.zeros((self.n_actions, self.action_size), dtype=np.float32)

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
        self.last_actions[:] = 0.0
        obs = np.concatenate([obs, self.last_actions.flatten()])

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


        self.last_actions = np.roll(self.last_actions, 1, 0)
        self.last_actions[0] = action_vec
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.concatenate([obs, self.last_actions.flatten()])

        return obs, reward, terminated, truncated, info
    


class InfoActionHistoryWrapper(Wrapper):
    def __init__(self, env, var_names, n_actions):
        super().__init__(env)
        self.var_names = var_names
        self.n_actions = n_actions

        # Amostra para descobrir dimensão da ação
        sample = self.env.action_space.sample()
        self.action_size = sample.size if isinstance(sample, np.ndarray) else 1

        # Buffer das últimas n_actions
        self.last_actions = np.zeros((self.n_actions, self.action_size), dtype=np.float32)

        # Espaço de observação: |vars| + n_actions * action_size
        obs_dim = len(self.var_names)
        total_dim = obs_dim + self.n_actions * self.action_size
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_dim,), dtype=np.float32
        )
        # Reward/Action/Info spaces seguem do env original

    def reset(self, **kwargs):
        # Lida com Gym >=0.26 (obs, info) e Gym <0.26 (obs)
        obs, info = self.env.reset(**kwargs)

        # Zera histórico de ações
        self.last_actions[:] = 0.0

        # Concatena obs de info + histórico de ações
        obs = self._extract(info)
        return np.concatenate([obs, self.last_actions.flatten()]), info

    def step(self, action):
        # Atualiza buffer de ações
        if isinstance(action, (int, np.integer)):
            action_vec = np.array([action], dtype=np.float32)
        else:
            action_vec = action

        self.last_actions = np.roll(self.last_actions, 1, 0)
        self.last_actions[0] = action_vec

        # Executa o step no env original
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Concatena obs de info + histórico de ações
        obs = self._extract(info)
        return np.concatenate([obs, self.last_actions.flatten()]), reward, terminated, truncated, info

    def _extract(self, info):
        # Monta vetor [ info[var] for var in var_names ]
        vals = [ info.get(var, 0.0) for var in self.var_names ]
        return np.array(vals, dtype=np.float32)