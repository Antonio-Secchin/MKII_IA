import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import os

from stable_baselines3.common.callbacks import BaseCallback

#TODO Fazer o callback do grafico, falta fazer um processamento nos dados para ficar uma reta

class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (gym.Env) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    """

    def __init__(self, eval_env, save_dir,n_eval_episodes=3, eval_freq=10, generate_graphic = False,  env_info = None):
        super().__init__()
        self._eval_env = eval_env
        self._n_eval_episodes = n_eval_episodes
        self._eval_freq = eval_freq
        self._best_mean_reward = -np.inf
        self._save_path = os.path.join(save_dir, "newest_model")
        self._graph_path = os.path.join(save_dir, "reward_plot.png")
        self._save_path_models_rollout = os.path.join(save_dir, "RolloutModels/")
        self._rollouts_done = 0
        self._rollout_reward = 0
        self._rollout_rewards_value = []
        self._means_rewards = []
        self._generate_graphic = generate_graphic
        self.infos(save_dir, env_info)

    def _on_step(self) -> bool:
        self._rollout_reward += self.locals["rewards"]
        return True

    def _on_rollout_end(self) -> None:
        """
        This method will be called by the model.

        :return: (bool)
        """

        self._rollouts_done += 1
        if self._rollouts_done % self._eval_freq != 0:
            self._rollout_reward = 0
            return  # Só avalia a cada `eval_freq` rollouts

        self._rollout_rewards_value.append(self._rollout_reward)
        self._rollout_reward = 0
        reward_sum = 0
        for _ in range(self._n_eval_episodes):
            obs = self._eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, info = self._eval_env.step(action)
                reward_sum += reward
        reward_mean = reward_sum/self._n_eval_episodes

        if reward_mean > self._best_mean_reward:
            print("New Reward Best Mean:", reward_mean)
            self._best_mean_reward = reward_mean
            self.model.save(self._save_path)

        if self._generate_graphic:
            self._means_rewards.append(reward_mean)

        filename = os.path.join(self._save_path_models_rollout, f"Model_Rollout_{self._rollouts_done}.zip")
        self.model.save(filename)
        print(f"✅ Modelo salvo em {filename}")

    def _on_training_end(self) -> None:
        # filename = os.path.join(self._save_path_models_rollout, "RewardsTrain.csv")
        # with open(filename, "w") as f:
        #     f.write("rollout,reward\n")
        #     for i, r in enumerate(self._rollout_rewards_value):
        #         f.write(f"{(i+1) * self._eval_freq},{int(r)}\n")
        if self._generate_graphic and not self._means_rewards:
            print("Nenhuma avaliação registrada. Gráfico não será gerado.")
            return

        x = np.array([(i + 1) * self._eval_freq for i in range(len(self._means_rewards))])
        y = np.array(self._means_rewards)

        plt.figure(figsize=(10, 6))

        if len(x) >= 4:  # spline cúbica requer pelo menos 4 pontos
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=3)
            y_smooth = spline(x_smooth)
            plt.plot(x_smooth, y_smooth, label="Recompensa Média (Suavizada)", linewidth=2)
        else:
            # poucos dados, plota direto
            plt.plot(x, y, marker='o', linestyle='-', label="Recompensa Média")

        plt.title("Evolução da Recompensa Média")
        plt.xlabel("Rollouts Realizados")
        plt.ylabel("Recompensa Média")
        plt.grid(True)
        plt.legend()
        plt.savefig(self._graph_path)
        plt.close()

        print(f"Gráfico de desempenho salvo em: {self._graph_path}")

    def callback_info(self):
        return {
            "n_eval_episodes": self._n_eval_episodes,
            "eval_freq": self._eval_freq
        }

    def infos(self, save_dir, env_info=None):
        # Monta o caminho do arquivo
        filepath = os.path.join(save_dir, "informacao_do_treino.txt")

        # Obtém as infos do próprio callback
        info_dict = self.callback_info()

        # Se o usuário quiser incluir info do ambiente
        if env_info is not None:
            info_dict["env_info"] = env_info

        # Escreve no arquivo TXT
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== INFORMAÇÕES DO TREINO ===\n\n")
            for key, value in info_dict.items():
                f.write(f"{key}: {value}\n")
    
class SimpleEvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (gym.Env) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    """

    def __init__(self, eval_env, save_dir,n_eval_episodes=5, eval_freq=100, generate_graphic = False, env_info = None):
        super().__init__()
        self._eval_env = eval_env
        self._n_eval_episodes = n_eval_episodes
        self._eval_freq = eval_freq
        self._best_mean_reward = -np.inf
        self._save_path = os.path.join(save_dir, "newest_model")
        self._graph_path = os.path.join(save_dir, "reward_plot.png")
        self._save_path_models_rollout = os.path.join(save_dir, "RolloutModels/")
        self._rollouts_done = 0
        self._rollout_reward = 0
        self._rollout_rewards_value = []
        self._means_rewards = []
        self._generate_graphic = generate_graphic
        self.infos(save_dir, env_info)

    def _on_step(self) -> bool:
        self._rollout_reward += self.locals["rewards"]
        return True

    def _on_rollout_end(self) -> None:
        """
        This method will be called by the model.

        :return: (bool)
        """

        self._rollouts_done += 1
        if self._rollouts_done % self._eval_freq != 0:
            self._rollout_reward = 0
            return  # Só avalia a cada `eval_freq` rollouts

        self._rollout_rewards_value.append(self._rollout_reward)
        self._rollout_reward = 0
        reward_sum = 0
        for _ in range(self._n_eval_episodes):
            obs = self._eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, info = self._eval_env.step(action)
                reward_sum += reward
        reward_mean = reward_sum/self._n_eval_episodes

        if reward_mean >= self._best_mean_reward:
            print("New Reward Best Mean:", reward_mean)
            self._best_mean_reward = reward_mean
            filename_best_model = os.path.join(self._save_path, "BestModel.zip")
            self.model.save(filename_best_model)

        if self._generate_graphic:
            self._means_rewards.append(reward_mean)

        filename = os.path.join(self._save_path_models_rollout, "LastModel.zip")
        self.model.save(filename)
        print(f"✅ Modelo salvo em {filename}")

    def _on_training_end(self) -> None:
        # filename = os.path.join(self._save_path_models_rollout, "RewardsTrain.csv")
        # with open(filename, "w") as f:
        #     f.write("rollout,reward\n")
        #     for i, r in enumerate(self._rollout_rewards_value):
        #         f.write(f"{(i+1) * self._eval_freq},{int(r)}\n")
        if self._generate_graphic or not self._means_rewards:
            print("Nenhuma avaliação registrada. Gráfico não será gerado.")
            return

        x = np.array([(i + 1) * self._eval_freq for i in range(len(self._means_rewards))])
        y = np.array(self._means_rewards)

        plt.figure(figsize=(10, 6))

        if len(x) >= 4:  # spline cúbica requer pelo menos 4 pontos
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=3)
            y_smooth = spline(x_smooth)
            plt.plot(x_smooth, y_smooth, label="Recompensa Média (Suavizada)", linewidth=2)
        else:
            # poucos dados, plota direto
            plt.plot(x, y, marker='o', linestyle='-', label="Recompensa Média")

        plt.title("Evolução da Recompensa Média")
        plt.xlabel("Rollouts Realizados")
        plt.ylabel("Recompensa Média")
        plt.grid(True)
        plt.legend()
        plt.savefig(self._graph_path)
        plt.close()

        print(f"Gráfico de desempenho salvo em: {self._graph_path}")

    def callback_info(self):
        return {
            "n_eval_episodes": self._n_eval_episodes,
            "eval_freq": self._eval_freq
        }
    
    def infos(self, save_dir, env_info=None):
        # Monta o caminho do arquivo
        filepath = os.path.join(save_dir, "informacao_do_treino.txt")

        # Obtém as infos do próprio callback
        info_dict = self.callback_info()

        # Se o usuário quiser incluir info do ambiente
        if env_info is not None:
            info_dict["env_info"] = env_info

        # Escreve no arquivo TXT
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== INFORMAÇÕES DO TREINO ===\n\n")
            for key, value in info_dict.items():
                f.write(f"{key}: {value}\n")