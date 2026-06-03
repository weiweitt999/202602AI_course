import os
import gymnasium as gym
import ale_py

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def make_env():
    env = gym.make("ALE/Breakout-v5")
    env = AtariWrapper(env)
    env = Monitor(env)
    return env

# Create Atari environment
env = DummyVecEnv([make_env])

# Stack 4 frames for Atari
env = VecFrameStack(env, n_stack=4)

# Create PPO model with CNN policy
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=0.0005,
    n_steps=128,
    batch_size=256,
    n_epochs=4
)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save("models/ppo_breakout_lr0005_100000")

env.close()

print("Atari Breakout training finished!")