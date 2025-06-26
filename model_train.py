import sys
import retro
from retro import Observations

import gymnasium as gym

import numpy as np

from stable_baselines3 import PPO,A2C, DQN

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

from gymnasium import Wrapper

from model_callback import EvalCallback

from env_wrappers import LastActionsWrapper, InfoActionHistoryWrapper

#TODO Fazer o wrapper que envolve as acoes nas observacoes
#### Game States: #####
###
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P')
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', render_mode = None)
###

#Usar players=<n> para colocar mais de um jogador/agente
# rew will be a list of [player_1_rew, player_2_rew]
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

def make_eval_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.RAM, render_mode = None)
    return env

# Ambiente para EXECUÇÃO com renderização
def make_render_env(var_names):
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.IMAGE, render_mode="human")
    wraper_env = InfoActionHistoryWrapper(env, var_names=var_names, n_actions=10)
    return wraper_env


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
info_env = make_info_env(variables)
vec_env = DummyVecEnv([lambda:info_env])


# Adiciona VecFrameStack (ex: 4 frames empilhados)
# Obs: 4 frames empilhados eh muito para o notebook (talvez o processador ou 16gb de RAM)
#stacked_env = VecFrameStack(vec_env, n_stack=8, channels_order='last')

# Treinar o modelo
#LstmPolicy
eval_callback = EvalCallback(eval_env=vec_env, save_dir = "Models", generate_graphic=True, eval_freq=10)

model = None
if len(sys.argv) > 1:
    if sys.argv[1] == "load":
        path = sys.argv[2]
        model = PPO.load(path, vec_env)
else:
    #DQN precisa mudar o action space entao provavelmente teria que fazer uma conexao entre um espaco discreto e o multi_binary
    model = PPO("MlpPolicy", vec_env, verbose=0)
#model.load("Models/newest_model.zip")
model.learn(total_timesteps=1000_000, progress_bar=True, callback= eval_callback)
vec_env.close()

#eval_env = DummyVecEnv([make_render_env])
# eval_env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.RAM, render_mode="human")

# obs = env.reset()
# print(env.action_space.shape)
# # Loop de execução com render
# while True:
#     #print("obs:", obs)
#     #action, _ = model.predict(obs, deterministic=False)
#     obs, reward, done, truncated, info = env.step(env.action_space.sample())
#     print(info)
#     print(obs.shape)
#     if done:  # `done` é uma lista/vetor no DummyVecEnv
#         obs = env.reset()