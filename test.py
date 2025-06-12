import sys
import retro
from retro import Observations

import gymnasium as gym

import numpy as np

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

from gymnasium import Wrapper

from model_callback import EvalCallback

#TODO Fazer o wrapper que envolve as acoes nas observacoes
#### Game States: #####
###
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P')
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', render_mode = None)
###

#Usar players=<n> para colocar mais de um jogador/agente
# rew will be a list of [player_1_rew, player_2_rew]
def make_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.IMAGE, render_mode = None)
    return env

def make_eval_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.RAM, render_mode = None)
    return env

# Ambiente para EXECUÇÃO com renderização
def make_render_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.RAM, render_mode="human")
    return env


env =make_render_env()
sample = env.action_space.sample() # [1 1 1 0 0 0 0 0 1 1 0 0]
print(sample)

obs = env.reset()
# Loop de execução com render
while True:
    #print("obs:", obs)
    #action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, truncated, info = env.step(sample)
    if done:  # `done` é uma lista/vetor no DummyVecEnv
        obs = env.reset()
# #Torna o ambiente vetorizado (requerido por SB3)
# vec_env = DummyVecEnv([make_env])

# # Adiciona VecFrameStack (ex: 4 frames empilhados)
# # Obs: 4 frames empilhados eh muito para o notebook (talvez o processador ou 16gb de RAM)
# stacked_env = VecFrameStack(vec_env, n_stack=2, channels_order='last')

# # Treinar o modelo
# #LstmPolicy
# eval_callback = EvalCallback(eval_env=stacked_env, save_dir = "Models", generate_graphic=True)

# model = PPO("MlpPolicy", stacked_env, verbose=0)
# model.learn(total_timesteps=20_000, progress_bar=True, callback= eval_callback)
# stacked_env.close()
