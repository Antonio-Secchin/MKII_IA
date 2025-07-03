import sys
import retro
from retro import Observations

import gymnasium as gym

import numpy as np

from stable_baselines3 import PPO, A2C, DQN

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

from gymnasium import Wrapper

from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from model_callback import EvalCallback

from env_wrappers import LastActionsWrapper, InfoActionHistoryWrapper

import multiprocessing


#### Game States: #####
###
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P')
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', render_mode = None)
###

#Usar players=<n> para colocar mais de um jogador/agente
# rew will be a list of [player_1_rew, player_2_rew]

def make_info_env_fn(var_names, seed=0):
    def _init():
        env = retro.make(
            game='MortalKombatII-Genesis',
            state='Level1.LiuKangVsJax',
            obs_type=Observations.RAM,
            render_mode=None
        )
        wrapped_env = InfoActionHistoryWrapper(env, var_names=var_names, n_actions=10)
        #return MaxAndSkipEnv(wrapped_env,12)
        return wrapped_env
    return _init

def make_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', obs_type=Observations.RAM, render_mode = None)
    wraper_env = LastActionsWrapper(env, n_actions=10)
    return wraper_env

def make_env_image():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', obs_type=Observations.IMAGE, render_mode = None)
    return env

def make_info_env(var_names):
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', obs_type=Observations.RAM, render_mode = None)
    wraper_env = InfoActionHistoryWrapper(env, var_names=var_names, n_actions=10)
    return wraper_env

def make_eval_env(var_names):
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', obs_type=Observations.RAM, render_mode = None)
    wraper_env = InfoActionHistoryWrapper(env, var_names=var_names, n_actions=10)
    return wraper_env

# Ambiente para EXECUÇÃO com renderização
def make_render_env(var_names):
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.RAM, render_mode="human")
    wraper_env = InfoActionHistoryWrapper(env, var_names=var_names, n_actions=10)
    return wraper_env


if __name__ == "__main__":

    #print(multiprocessing.cpu_count())

    variables = [
        "Tempo",
        "health",
        "enemy_health",
        "rounds_won",
        "enemy_rounds_won",
        "wins",
        "x_position",
        "y_position",
        "enemy_x_position",
        "enemy_y_position"
    ]
        

    #Torna o ambiente vetorizado (requerido por SB3)
    #vec_env = DummyVecEnv([make_env])
    #info_env = MaxAndSkipEnv(make_info_env(variables),12)
    eval_env = make_eval_env(variables)
    vec_env = DummyVecEnv([lambda:eval_env])

    # Testando paralelismo
    num_envs = 8  # ou quantos sua CPU suportar
    env_fns = [make_info_env_fn(variables, seed=i) for i in range(num_envs)]

    vec_env_para = SubprocVecEnv(env_fns)

    # Adiciona VecFrameStack (ex: 4 frames empilhados)
    # Obs: 4 frames empilhados eh muito para o notebook (talvez o processador ou 16gb de RAM)
    #stacked_env = VecFrameStack(vec_env, n_stack=8, channels_order='last')

    # Treinar o modelo
    eval_callback = EvalCallback(eval_env=vec_env, save_dir = "Models", generate_graphic=True, eval_freq=20, n_eval_episodes=10)

    model = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "load":
            path = sys.argv[2]
            model = PPO.load(path, vec_env)
    else:
        model = PPO("MlpPolicy", vec_env, verbose=0, device='cpu')
    model.learn(total_timesteps=200_000, progress_bar=True, callback= eval_callback)
    print("Qtd de calls do wrapper.step: ", eval_env.n_steps)
    vec_env.close()