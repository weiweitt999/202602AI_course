import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np

# Load trained DQN model
model = DQN.load("models/dqn_cartpole_50000")

# Create environment
env = gym.make("CartPole-v1")

episode_rewards = []

# Evaluate for 10 episodes
for episode in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: reward = {total_reward}")

env.close()

print("Average reward:", np.mean(episode_rewards))
print("Std reward:", np.std(episode_rewards))