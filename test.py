import sys
import retro

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv

# Ambiente para TREINAMENTO sem render
#Testar com VecFrameStack: it stacks several consecutive observations
train_env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', render_mode=None)
train_env = DummyVecEnv([lambda: train_env])

obs = train_env.reset()
print(obs.shape)
#sys.exit(0)

# Treinar o modelo
#LstmPolicy
model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=15_000)
train_env.close()

# Ambiente para EXECUÇÃO com renderização
def make_render_env():
    env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P', render_mode="human")
    return env

eval_env = DummyVecEnv([make_render_env])
obs = eval_env.reset()

# Loop de execução com render
while True:
    #print("obs:", obs)
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = eval_env.step(action)

    if done[0]:  # `done` é uma lista/vetor no DummyVecEnv
        obs = eval_env.reset()

#env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P')
# env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax', render_mode = None)

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)
# #obs = env.reset()

# vec_env = model.get_env()
# obs = vec_env.reset()

# while True:

#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")

#     if done:
#         obs = env.reset()
