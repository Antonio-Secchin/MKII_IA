import sys
import retro
from retro import Observations

import gymnasium as gym

import numpy as np

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

from gymnasium import Wrapper

from model_callback import EvalCallback

from env_wrappers import LastActionsWrapper

#TODO Fazer o wrapper que envolve as acoes nas observacoes
#### Game States: #####
###
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P')
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', render_mode = None)
###

#Usar players=<n> para colocar mais de um jogador/agente
# rew will be a list of [player_1_rew, player_2_rew]
def make_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.RAM, render_mode = None)
    wraper_env = LastActionsWrapper(env, 10)
    return wraper_env

def make_eval_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.RAM, render_mode = None)
    return env

# Ambiente para EXECUÇÃO com renderização
def make_render_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.RAM, render_mode="human")
    return env

    

#Torna o ambiente vetorizado (requerido por SB3)
vec_env = DummyVecEnv([make_env])

# Adiciona VecFrameStack (ex: 4 frames empilhados)
# Obs: 4 frames empilhados eh muito para o notebook (talvez o processador ou 16gb de RAM)
stacked_env = VecFrameStack(vec_env, n_stack=8, channels_order='last')

# Treinar o modelo
#LstmPolicy
eval_callback = EvalCallback(eval_env=stacked_env, save_dir = "Models", generate_graphic=True)

model = PPO("MlpPolicy", stacked_env, verbose=0)
model.learn(total_timesteps=100_000, progress_bar=True, callback= eval_callback)
stacked_env.close()

#eval_env = DummyVecEnv([make_render_env])
# eval_env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.RAM, render_mode="human")

# obs = eval_env.reset()
# print(eval_env.action_space.shape)
# # Loop de execução com render
# while True:
#     #print("obs:", obs)
#     #action, _ = model.predict(obs, deterministic=False)
#     obs, reward, done, truncated, info = eval_env.step(eval_env.action_space.sample())
#     print(obs)
#     print(obs.shape)
#     if done:  # `done` é uma lista/vetor no DummyVecEnv
#         obs = eval_env.reset()