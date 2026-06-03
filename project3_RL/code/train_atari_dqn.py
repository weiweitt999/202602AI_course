import os
import gymnasium as gym
import ale_py

from stable_baselines3 import DQN
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

# Stack 4 frames, common setting for Atari
env = VecFrameStack(env, n_stack=4)

# Create DQN model
model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    buffer_size=50000,
    learning_starts=10000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save("models/dqn_breakout_100000")

env.close()

print("DQN Atari Breakout training finished! 100000 steps")