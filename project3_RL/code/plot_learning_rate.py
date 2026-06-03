import os
import pandas as pd
import matplotlib.pyplot as plt

# Create results folder
os.makedirs("results", exist_ok=True)

# Learning rate experiment results
data = {
    "Learning Rate": [0.0001, 0.0003, 0.001],
    "Average Reward": [244.4, 292.2, 491.7],
    "Std": [61.10, 104.02, 24.9]
}

df = pd.DataFrame(data)

# Save table as CSV
df.to_csv("results/learning_rate_results.csv", index=False)

# Plot learning rate result
plt.figure(figsize=(7, 5))

plt.errorbar(
    df["Learning Rate"],
    df["Average Reward"],
    yerr=df["Std"],
    marker="o",
    capsize=5,
    linewidth=2
)

plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Average Reward")
plt.title("Effect of Learning Rate on PPO Performance in CartPole-v1")
plt.grid(True)

# Save figure
plt.savefig("results/learning_rate_reward.png", dpi=300, bbox_inches="tight")

# Show figure
plt.show()

print("Saved results/learning_rate_results.csv")
print("Saved results/learning_rate_reward.png")