import gymnasium as gym
import ale_py

env = gym.make("ALE/Breakout-v5")

obs, info = env.reset()

print("Atari environment created successfully!")
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

env.close()