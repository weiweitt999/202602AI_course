import gymnasium as gym
import ale_py
import numpy as np

from stable_baselines3.common.atari_wrappers import AtariWrapper

# Create Atari Breakout environment
env = gym.make("ALE/Breakout-v5")
env = AtariWrapper(env)

episode_rewards = []

# Evaluate random agent for 10 episodes
for episode in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Randomly sample an action from the action space
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: reward = {total_reward}")

env.close()

print("Random Agent Average Reward:", np.mean(episode_rewards))
print("Random Agent Std:", np.std(episode_rewards))