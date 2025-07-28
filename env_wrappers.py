import sys
import retro
from retro import Observations

import gymnasium as gym

import numpy as np

from gymnasium import Wrapper

NoAction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

FogoBaixo = [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

Voadora = [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]

#5 segundos
Chute_Bicicleta = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

#Bola de Fogo Alta : Frente, Frente, Soco Alto, No emulador nao tem soco alto por controle

class LastActionsWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, n_actions, steps_between_actions = 11):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)
        self.n_actions = n_actions
        self.steps_between_actions = steps_between_actions

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
    def __init__(self, env, var_names, n_actions, steps_between_actions = 11):
        super().__init__(env)
        self.var_names = var_names
        self.n_actions = n_actions
        self.steps_between_actions = steps_between_actions

        # Amostra para descobrir dimensão da ação
        sample = self.env.action_space.sample()
        self.action_size = sample.size if isinstance(sample, np.ndarray) else 1

        # Buffer das últimas n_actions
        self.last_actions = np.zeros((self.n_actions, self.action_size), dtype=np.float32)
        self.n_steps = 0
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
        self.n_steps += 1
        if isinstance(action, (int, np.integer)):
            action_vec = np.array([action], dtype=np.float32)
        else:
            action_vec = action

        self.last_actions = np.roll(self.last_actions, 1, 0)
        self.last_actions[0] = action_vec
        
        total_reward = 0
        # Executa o step no env original
        obs, reward, terminated, truncated, info = self.env.step(action)
        total_reward += reward

        for _ in range(self.steps_between_actions):
            obs, reward, terminated, truncated, info = self.env.step(NoAction)
            total_reward += reward
            if terminated or truncated:
                break
        # Concatena obs de info + histórico de ações
        obs = self._extract(info)
        return np.concatenate([obs, self.last_actions.flatten()]), total_reward, terminated, truncated, info

    def _extract(self, info):
        # Monta vetor [ info[var] for var in var_names ]
        vals = [ info.get(var, 0.0) for var in self.var_names ]
        return np.array(vals, dtype=np.float32)
    

#Wrapper para testar as actions compostas
class TestActionWrapper(Wrapper):
    def __init__(self, env, var_names, n_actions, steps_between_actions = 11):
        super().__init__(env)
        self.var_names = var_names
        self.n_actions = n_actions
        self.steps_between_actions = steps_between_actions

        # Amostra para descobrir dimensão da ação
        sample = self.env.action_space.sample()
        self.action_size = sample.size + 3 if isinstance(sample, np.ndarray) else 1

        # Buffer das últimas n_actions
        self.last_actions = np.zeros((self.n_actions, self.action_size), dtype=np.float32)

        # Espaço de observação: |vars| + n_actions * action_size

        #Ver esse calculo de total_dim
        obs_dim = len(self.var_names)
        total_dim = obs_dim + self.n_actions * self.action_size
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_dim,), dtype=np.float32
        )
        #TODO: Ver essa seed
        self.action_space = gym.spaces.MultiBinary(self.action_size)
        # Reward/Action/Info spaces seguem do env original
        self._iskicking = False
        self._kicktimer = 0

    def reset(self, **kwargs):
        # Lida com Gym >=0.26 (obs, info) e Gym <0.26 (obs)
        obs, info = self.env.reset(**kwargs)

        # Zera histórico de ações
        self.last_actions[:] = 0.0
        self._iskicking = False
        self._kicktimer = 0
        # Concatena obs de info + histórico de ações
        obs = self._extract(info)
        return np.concatenate([obs, self.last_actions.flatten()]), info

    def step(self, action):
        # Atualiza buffer de ações
        if isinstance(action, (int, np.integer)):
            action_vec = np.array([action], dtype=np.float32)
        else:
            action_vec = action

    
        sum_actions = np.sum(action)

        total_reward = 0
        if(action[-3] == 1 and sum_actions ==1):
            self._iskicking = True

        if(action[-1] == 1 and sum_actions ==1):
            if self._iskicking:
                self._iskicking = False
                self._kicktimer = 0

            #Executa um golpe especial Voadora
            for act in Voadora:
                obs, reward, terminated, truncated, info = self.env.step(act)
                total_reward += reward
                if terminated or truncated:
                    break
                for _ in range(self.steps_between_actions):
                    obs, reward, terminated, truncated, info = self.env.step(NoAction)
                    total_reward += reward
                    if terminated or truncated:
                        break
        
        elif(action[-2] == 1 and sum_actions ==1):
            if self._iskicking:
                self._iskicking = False
                self._kicktimer = 0
            #Executa um golpe especial Fogo Baixo
            for act in FogoBaixo:
                obs, reward, terminated, truncated, info = self.env.step(act)
                total_reward += reward
                if terminated or truncated:
                    break
                for _ in range(self.steps_between_actions):
                    obs, reward, terminated, truncated, info = self.env.step(NoAction)
                    total_reward += reward
                    if terminated or truncated:
                        break

        
        else:
            # Executa o step no env original
            #TODO Ver como tratar o caso de ter acao especial com acao normal ex:[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
            if np.sum(action[12:]) != 0:
                action = NoAction
            #Isso pega os 12 primeiros ne?
            if self._iskicking and action[1] != 1 and np.sum(action[7:11]) == 0 and self._kicktimer < 231:
                self._kicktimer += 1
                action[0] = 1
                action[-3] = 1
                action_vec[-3] = 1
            else:
                self._kicktimer = 0
                self._iskicking = 0
            action = action[:12]
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            for _ in range(self.steps_between_actions):
                obs, reward, terminated, truncated, info = self.env.step(NoAction)
                total_reward += reward
                if terminated or truncated:
                    break
        
        self.last_actions = np.roll(self.last_actions, 1, 0)
        self.last_actions[0] = action_vec
        # Concatena obs de info + histórico de ações
        obs = self._extract(info)
        return np.concatenate([obs, self.last_actions.flatten()]), total_reward, terminated, truncated, info

    #Fazer a verificacao do -1 para ver se esta defendendo
    def _extract(self, info):
        # Monta vetor [ info[var] for var in var_names ]
        #Verificando se esta defendendo
        if info["Block_enemy"] == -1:
            info["Block_enemy"] = 1
        else:
            info["Block_enemy"] = 0
        
        if info["Block_aliado"] == -1:
            info["Block_aliado"] = 1
        else:
            info["Block_aliado"] = 0
        
        vals = [ info.get(var, 0.0) for var in self.var_names ]
        return np.array(vals, dtype=np.float32)