import sys
import retro

import gymnasium as gym

import numpy as np

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

from gymnasium import Wrapper

from model_callback import EvalCallback


#### Game States: #####
###
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P')
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', render_mode = None)
###

def make_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', render_mode = None)
    return env

def make_eval_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', render_mode = None)
    return env

# Ambiente para EXECUÇÃO com renderização
def make_render_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', render_mode="human")
    return env

    

# Torna o ambiente vetorizado (requerido por SB3)
vec_env = DummyVecEnv([make_env])

# Adiciona VecFrameStack (ex: 4 frames empilhados)
stacked_env = VecFrameStack(vec_env, n_stack=4, channels_order='last')

# Treinar o modelo
#LstmPolicy
eval_callback = EvalCallback(eval_env=stacked_env, save_dir = "Models")

model = PPO("MlpPolicy", stacked_env, verbose=0)
model.learn(total_timesteps=30_000, progress_bar=True, callback= eval_callback)
stacked_env.close()



eval_env = DummyVecEnv([make_render_env])
obs = eval_env.reset()

# Loop de execução com render
while True:
    #print("obs:", obs)
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = eval_env.step(action)

    if done[0]:  # `done` é uma lista/vetor no DummyVecEnv
        obs = eval_env.reset()
