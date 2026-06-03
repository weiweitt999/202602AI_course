import gymnasium as gym
import ale_py
import numpy as np
from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def make_env():
    env = gym.make("ALE/Breakout-v5")
    env = AtariWrapper(env)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

model = PPO.load("models/ppo_breakout_lr0005_100000")

episode_rewards = []

for episode in range(10):
    obs = env.reset()
    done = [False]
    total_reward = 0

    while not done[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: reward = {total_reward}")

env.close()

print("Average reward:", np.mean(episode_rewards))
print("Std reward:", np.std(episode_rewards))