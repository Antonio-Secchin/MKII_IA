import sys
import retro
from retro import Observations

import gymnasium as gym

import numpy as np

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

from gymnasium import Wrapper

from model_callback import EvalCallback

import time

'''
['B']
['A']
[]
['START']
['UP']
['DOWN']
['LEFT']
['RIGHT']
['C']
['Y']
['X']
['Z']
'''

#TODO Fazer o wrapper que envolve as acoes nas observacoes
#### Game States: #####
###
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P')
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', render_mode = None)
###

#Usar players=<n> para colocar mais de um jogador/agente
# rew will be a list of [player_1_rew, player_2_rew]
def make_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.IMAGE, render_mode = None, use_restricted_actions= retro.Actions.FILTERED)
    return env

def make_eval_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.RAM, render_mode = None)
    return env

# Ambiente para EXECUÇÃO com renderização
def make_render_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', obs_type=Observations.RAM, render_mode="human")
    return env

target_fps = 30
frame_time = 1.0 / target_fps

env = make_render_env()
sample = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#env.action_space.sample() # [1 1 1 0 0 0 0 0 1 1 0 0]
actions = []
# for p, action in enumerate(env.action_to_array(sample)):
#     actions.append(
#         [env.buttons[i] for i in np.extract(action, np.arange(len(action)))],
#     )
#     print(actions)
# print(sample)
print(env.get_action_meaning(sample))

# for _ in range(13):
#     print(env.get_action_meaning(sample))
#     sample = np.roll(sample,1,0)

obs = env.reset()
# Loop de execução com render
while True:
    start_time = time.time()
    #print("obs:", obs)
    #action, _ = model.predict(obs, deterministic=False)
    sample = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(sample)
    if reward>0:
        print(reward)
        print(env.get_action_meaning(sample))
    if done:  # `done` é uma lista/vetor no DummyVecEnv
        obs = env.reset()
    # elapsed = time.time() - start_time
    # sleep_time = max(0.0, frame_time - elapsed)
    # time.sleep(sleep_time)
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
