# plot_env_comparison.py

import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# Data for environment comparison
environments = ["CartPole-v1\nPPO 50k", "Breakout-v5\nPPO 200k"]
average_rewards = [500.0, 0.4]
std_rewards = [0.0, 1.2]

plt.figure(figsize=(7, 5))

plt.bar(environments, average_rewards)

plt.errorbar(
    environments,
    average_rewards,
    yerr=std_rewards,
    fmt="none",
    capsize=5
)

plt.xlabel("Environment")
plt.ylabel("Average Reward")
plt.title("PPO Performance Comparison between CartPole-v1 and Atari Breakout-v5")
plt.grid(axis="y")

plt.savefig(
    "results/fig7_cartpole_vs_atari.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved results/fig7_cartpole_vs_atari.png")