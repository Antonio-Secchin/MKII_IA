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
# retro.make(game='MortalKombatII-Genesis', state='LiuKangVsScorpion_VeryHard_11', obs_type=Observations.IMAGE, render_mode = None)
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
        wrapped_env = TestActionWrapper(env, var_names=var_names, n_actions=10)
        return wrapped_env
    return _init

def make_env(var_names):
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', obs_type=Observations.RAM, render_mode = None)
    #wraper_env = InfoActionHistoryWrapper(env, n_actions=10)
    return env

def make_env_image(var_names):
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', obs_type=Observations.IMAGE, render_mode = None)
    return env

def make_info_env(var_names):
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', obs_type=Observations.RAM, render_mode = None)
    wraper_env = InfoActionHistoryWrapper(env, var_names=var_names, n_actions=10)
    return wraper_env

def make_test_env(var_names):
    env = retro.make(game='MortalKombatII-Genesis', state='LiuKangVsScorpion_VeryHard_11', obs_type=Observations.RAM, render_mode = None)
    wraper_env = TestActionWrapper(env, var_names=var_names, n_actions=10, steps_between_actions=11)
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
        "enemy_y_position",
        "Block_aliado",
        "Block_inimigo",
        "Projectile_x",
        "Projectile_y",
        "Projectile_Position_Enemy_x",
        "Projectile_Position_Enemy_y",
    ]
        

    #Torna o ambiente vetorizado (requerido por SB3)
    #vec_env = DummyVecEnv([make_env])
    eval_env = make_test_env(variables)
    #eval_env = make_env(variables)
    #eval_env = make_info_env(variables)
    #eval_env = make_env_image(variables)

    vec_env = DummyVecEnv([lambda:eval_env])

    # Testando paralelismo
    # num_envs = 8  # ou quantos sua CPU suportar
    # env_fns = [make_info_env_fn(variables, seed=i) for i in range(num_envs)]

    #vec_env_para = SubprocVecEnv(env_fns)

    # Adiciona VecFrameStack (ex: 4 frames empilhados)
    #stacked_env = VecFrameStack(vec_env, n_stack=4, channels_order='last')
    #env_info = "RAM_env \n Action_space: actions"
    # Treinar o modelo
    eval_callback = SimpleEvalCallback(eval_env=vec_env, save_dir = "Models/RAM_Red_Scorpion_att", generate_graphic=True, eval_freq=100, n_eval_episodes=20, env_info=eval_env.env_info())
    #eval_callback = SimpleEvalCallback(eval_env=stacked_env, save_dir = "Models/Image_env_stacked_Scorpion_att", generate_graphic=True, eval_freq=100, n_eval_episodes=20, env_info=env_info)

    #Mudar para treinar sem parar e ajustar para salvar o ultimo modelo e o melhor modelo nas ultimas n iterações
    model = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "load":
            path = sys.argv[2]
            model = PPO.load(path, vec_env)
    else:
        #model para imagem
        #model = PPO("CnnPolicy", stacked_env, verbose=0, device='cuda')
        model = PPO("MlpPolicy", vec_env, verbose=0, device='cpu')
    model.learn(total_timesteps=15_000_000, progress_bar=True, callback= eval_callback)
    vec_env.close()
