import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import os

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Create CartPole environment
env = gym.make("CartPole-v1")
env = Monitor(env)

# Create DQN model
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.001,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05
)

# Train DQN agent
model.learn(total_timesteps=50000)

# Save model
model.save("models/dqn_cartpole_50000")

env.close()

print("DQN CartPole training finished! 10000 steps")