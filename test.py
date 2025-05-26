import retro

#env = retro.make(game='MortalKombatII-Genesis', state='Level1.LiuKangVsJax.2P')
env = retro.make(game='MortalKombatII-Genesis', state='LiuKangVsJax_VeryHard_03')
obs = env.reset()

while True:
    env.render()

    # Ação aleatória (só para testar)
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    if done:
        obs = env.reset()
