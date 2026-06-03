import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# =====================================================
# Figure 1: CartPole training steps
# =====================================================
cartpole_steps = [10000, 50000]
cartpole_steps_rewards = [292.2, 500.0]
cartpole_steps_std = [104.02, 0.0]

plt.figure(figsize=(7, 5))
plt.errorbar(
    cartpole_steps,
    cartpole_steps_rewards,
    yerr=cartpole_steps_std,
    marker="o",
    capsize=5,
    linewidth=2
)
plt.xlabel("Training Steps")
plt.ylabel("Average Reward")
plt.title("Effect of Training Steps on PPO Performance in CartPole-v1")
plt.grid(True)
plt.savefig("results/fig1_cartpole_training_steps.png", dpi=300, bbox_inches="tight")
plt.show()


# =====================================================
# Figure 2: CartPole learning rate
# =====================================================
cartpole_lr = [0.0001, 0.0003, 0.001]
cartpole_lr_rewards = [244.4, 292.2, 491.7]
cartpole_lr_std = [61.10, 104.02, 24.9]

plt.figure(figsize=(7, 5))
plt.errorbar(
    cartpole_lr,
    cartpole_lr_rewards,
    yerr=cartpole_lr_std,
    marker="o",
    capsize=5,
    linewidth=2
)
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Average Reward")
plt.title("Effect of Learning Rate on PPO Performance in CartPole-v1")
plt.grid(True)
plt.savefig("results/fig2_cartpole_learning_rate.png", dpi=300, bbox_inches="tight")
plt.show()


# =====================================================
# Figure 3: Atari training steps
# =====================================================
atari_steps = [50000, 100000, 200000]
atari_steps_rewards = [0.4, 0.4, 0.4]
atari_steps_std = [1.2, 0.8, 1.2]

plt.figure(figsize=(7, 5))
plt.errorbar(
    atari_steps,
    atari_steps_rewards,
    yerr=atari_steps_std,
    marker="o",
    capsize=5,
    linewidth=2
)
plt.xlabel("Training Steps")
plt.ylabel("Average Reward")
plt.title("Effect of Training Steps on PPO Performance in Atari Breakout-v5")
plt.grid(True)
plt.savefig("results/fig3_atari_training_steps.png", dpi=300, bbox_inches="tight")
plt.show()


# =====================================================
# Figure 4: Atari learning rate
# =====================================================
atari_lr = [0.0001, 0.00025, 0.0005]
atari_lr_rewards = [0.5, 0.4, 1.4]
atari_lr_std = [0.806, 0.8, 2.20]

plt.figure(figsize=(7, 5))
plt.errorbar(
    atari_lr,
    atari_lr_rewards,
    yerr=atari_lr_std,
    marker="o",
    capsize=5,
    linewidth=2
)
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Average Reward")
plt.title("Effect of Learning Rate on PPO Performance in Atari Breakout-v5")
plt.grid(True)
plt.savefig("results/fig4_atari_learning_rate.png", dpi=300, bbox_inches="tight")
plt.show()


# =====================================================
# Figure 5: Random vs DQN vs PPO on Atari
# =====================================================
agents = ["Random", "DQN", "PPO"]
agent_rewards = [0.3, 0.9, 1.4]
agent_std = [0.46, 1.58, 2.20]

plt.figure(figsize=(7, 5))
plt.bar(agents, agent_rewards)
plt.errorbar(
    agents,
    agent_rewards,
    yerr=agent_std,
    fmt="none",
    capsize=5
)
plt.xlabel("Agent / Algorithm")
plt.ylabel("Average Reward")
plt.title("Random Agent vs DQN vs PPO on Atari Breakout-v5")
plt.grid(axis="y")
plt.savefig("results/fig5_atari_algorithm_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("All final figures saved in the results folder.")

# =====================================================
# Figure 6: PPO vs DQN on CartPole-v1
# =====================================================
labels = ["PPO\n10k", "DQN\n10k", "PPO\n50k", "DQN\n50k"]
rewards = [292.2, 185.6, 500.0, 90.5]
std = [104.02, 56.50, 0.0, 46.85]

plt.figure(figsize=(7, 5))
plt.bar(labels, rewards)
plt.errorbar(
    labels,
    rewards,
    yerr=std,
    fmt="none",
    capsize=5
)
plt.xlabel("Algorithm and Training Steps")
plt.ylabel("Average Reward")
plt.title("PPO vs DQN on CartPole-v1")
plt.grid(axis="y")
plt.savefig("results/fig6_cartpole_algorithm_comparison.png", dpi=300, bbox_inches="tight")
plt.show()