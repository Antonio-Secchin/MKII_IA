import sys
import retro
from retro import Observations

import gymnasium as gym

import numpy as np

from stable_baselines3 import PPO, A2C, DQN

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

from gymnasium import Wrapper

from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from model_callback import EvalCallback, SimpleEvalCallback

from env_wrappers import LastActionsWrapper, InfoActionHistoryWrapper, TestActionWrapper

import multiprocessing


#### Game States: #####
###
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P')
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', render_mode = None)
###

#Usar players=<n> para colocar mais de um jogador/agente
# action_space will by MultiBinary(16) now instead of MultiBinary(8)
# the bottom half of the actions will be for player 1 and the top half for
# rew will be a list of [player_1_rew, player_2_rew]
# done and info will remain the same

def make_test_env(var_names):
    env = retro.make(game='MortalKombatII-Genesis', state='LiuKangVsScorpion_VeryHard_11', obs_type=Observations.RAM, render_mode = None, players=2)
    wraper_env = TestActionWrapper(env, var_names=var_names, n_actions=10, steps_between_actions=11)
    
    return wraper_env