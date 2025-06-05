import matplotlib.pyplot as plt
import numpy as np
import os

from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm


class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (gym.Env) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    """

    def __init__(self, eval_env, save_dir,n_eval_episodes=5, eval_freq=5):
        super().__init__()
        self._eval_env = eval_env
        self._n_eval_episodes = n_eval_episodes
        self._eval_freq = eval_freq
        self._best_mean_reward = -np.inf
        self._save_path = os.path.join(save_dir, "newest_model")
        self._rollouts_done = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        """
        This method will be called by the model.

        :return: (bool)
        """

        self._rollouts_done += 1
        if self._rollouts_done % self._eval_freq != 0:
            return  # Só avalia a cada `eval_freq` rollouts

        if self.n_calls % self._eval_freq == 0:
            reward_sum = 0
            done = False
            for _ in range(self._n_eval_episodes):
                obs = self._eval_env.reset()
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self._eval_env.step(action)
                    reward_sum += reward

            if (reward_sum/self._n_eval_episodes) > self._best_mean_reward:
                self._best_mean_reward = reward_sum/self._n_eval_episodes
                self.model.save(self._save_path)

            print("Best mean reward: {:.2f}".format(float(self._best_mean_reward)))

    