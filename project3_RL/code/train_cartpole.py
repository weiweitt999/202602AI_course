import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

# Create folders for saving models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Create CartPole environment
env = gym.make("CartPole-v1")
env = Monitor(env)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001
)

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("models/ppo_cartpole_lr0001_10000")

env.close()

print("CartPole training finished!")