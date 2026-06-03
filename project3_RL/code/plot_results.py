import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# Experimental results
data = {
    "Environment": [
        "CartPole-v1",
        "CartPole-v1",
        "Breakout-v5",
        "Breakout-v5",
        "Breakout-v5"
    ],
    "Algorithm": [
        "PPO",
        "PPO",
        "PPO",
        "PPO",
        "PPO"
    ],
    "Training Steps": [
        10000,
        50000,
        50000,
        100000,
        200000
    ],
    "Average Reward": [
        292.2,
        500.0,
        0.4,
        0.4,
        0.4
    ],
    "Std": [
        104.02,
        0.0,
        1.2,
        0.8,
        1.2
    ]
}
df = pd.DataFrame(data)

# Save results table
df.to_csv("results/experiment_results.csv", index=False)

print("Experiment Results:")
print(df)

# Plot CartPole results
cartpole_df = df[df["Environment"] == "CartPole-v1"]

plt.figure()
plt.plot(
    cartpole_df["Training Steps"],
    cartpole_df["Average Reward"],
    marker="o"
)
plt.xlabel("Training Steps")
plt.ylabel("Average Reward")
plt.title("PPO Performance on CartPole-v1")
plt.grid(True)
plt.savefig("results/cartpole_reward.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot Breakout results
breakout_df = df[df["Environment"] == "Breakout-v5"]

plt.figure()
plt.plot(
    breakout_df["Training Steps"],
    breakout_df["Average Reward"],
    marker="o"
)
plt.xlabel("Training Steps")
plt.ylabel("Average Reward")
plt.title("PPO Performance on Atari Breakout-v5")
plt.grid(True)
plt.savefig("results/breakout_reward.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved:")
print("results/experiment_results.csv")
print("results/cartpole_reward.png")
print("results/breakout_reward.png")