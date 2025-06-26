import sys
import retro
from retro import Observations

from stable_baselines3 import PPO,A2C, DQN

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv, VecVideoRecorder


from env_wrappers import LastActionsWrapper, InfoActionHistoryWrapper

import imageio

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
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', obs_type=Observations.IMAGE, render_mode="rgb_array")
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
info_env = make_render_env(variables)


vec_env = DummyVecEnv([lambda:info_env])

# rec_env = VecVideoRecorder(
#     vec_env,
#     video_folder="./videos",
#     record_video_trigger=lambda step: step % 1000 == 0,
#     video_length=3400,
#     name_prefix="mk2_episode"
# )

model = None
if len(sys.argv) > 1:
    if sys.argv[1] == "load":
        path = sys.argv[2]
        model = PPO.load(path, vec_env)
else:
    model = PPO("MlpPolicy", vec_env, verbose=0)

frames = []
obs = vec_env.reset()
# Loop de execução com render
i = 0
#interessante o render mode human do DummyVecEnv
for _ in range(3):
    while True:
        frame = vec_env.render(mode="rgb_array")
        frames.append(frame)
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = vec_env.step(action)
        if done:  # `done` é uma lista/vetor no DummyVecEnv
            obs = vec_env.reset()
            break
vec_env.close()

# Grava como vídeo com qualidade personalizada
with imageio.get_writer("videos/video.mp4", fps=60, codec="libx264", bitrate="8000k") as writer:
    for frame in frames:
        writer.append_data(frame)