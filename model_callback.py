import matplotlib.pyplot as plt
import numpy as np
import os

from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

#TODO Fazer o callback do grafico, falta fazer um processamento nos dados para ficar uma reta

class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (gym.Env) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    """

    def __init__(self, eval_env, save_dir,n_eval_episodes=5, eval_freq=5, generate_graphic = False):
        super().__init__()
        self._eval_env = eval_env
        self._n_eval_episodes = n_eval_episodes
        self._eval_freq = eval_freq
        self._best_mean_reward = -np.inf
        self._save_path = os.path.join(save_dir, "newest_model")
        self._graph_path = os.path.join(save_dir, "reward_plot.png")
        self._rollouts_done = 0
        self._means_rewards = []
        self._generate_graphic = generate_graphic

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """
        This method will be called by the model.

        :return: (bool)
        """

        self._rollouts_done += 1
        if self._rollouts_done % self._eval_freq != 0:
            return  # Só avalia a cada `eval_freq` rollouts

        reward_sum = 0
        for _ in range(self._n_eval_episodes):
            obs = self._eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self._eval_env.step(action)
                reward_sum += reward
        reward_mean = reward_sum/self._n_eval_episodes
        if self._generate_graphic:
            self._means_rewards.append(reward_mean)

        if reward_mean > self._best_mean_reward:
            self._best_mean_reward = reward_mean
            self.model.save(self._save_path)

        print("Best mean reward: {:.2f}".format(float(self._best_mean_reward)))

    def _on_training_end(self) -> None:
        if self._generate_graphic and not self._means_rewards:
            print("Nenhuma avaliação registrada. Gráfico não será gerado.")
            return

        x = [(i + 1) * self._eval_freq for i in range(len(self._means_rewards))]

        plt.figure(figsize=(10, 6))
        plt.plot(x, self._means_rewards, marker='o', linestyle='-')
        plt.title("Evolução da Recompensa Média")
        plt.xlabel("Rollouts Realizados")
        plt.ylabel("Recompensa Média")
        plt.grid(True)
        plt.savefig(self._graph_path)
        plt.close()

        print(f"Gráfico de desempenho salvo em: {self._graph_path}")

    