# AI HW3: Reinforcement Learning Experiments

This project is for Artificial Intelligence HW3.
The goal is to train and evaluate reinforcement learning agents using Gymnasium environments.

Two environments are used in this project:

* **CartPole-v1**
* **Atari Breakout-v5**

The experiments compare the effects of:

* Training steps
* Learning rate
* Algorithm choice
* Random agent baseline

The reinforcement learning algorithms used are:

* **PPO (Proximal Policy Optimization)**
* **DQN (Deep Q-Network)**

---

## Project Structure

```text
project3_RL/
│
├── train_cartpole.py
├── eval_cartpole.py
├── train_cartpole_dqn.py
├── eval_cartpole_dqn.py
│
├── train_atari.py
├── eval_atari.py
├── train_atari_dqn.py
├── eval_atari_dqn.py
├── eval_random_atari.py
│
├── test_atari.py
│
├── plot_results.py
├── plot_learning_rate.py
├── plot_final_figures.py
│
├── models/
├── results/
└── README.md
```

---

## Environment Setup

Python 3.10 is recommended.

Create and activate a virtual environment:

```bash
python -m venv rl_hw3_310
```

For Windows PowerShell:

```bash
.\rl_hw3_310\Scripts\Activate
```

Install required packages:

```bash
pip install stable-baselines3[extra]
pip install "gymnasium[atari,accept-rom-license]"
pip install ale-py matplotlib pandas
```

---

## Files Description

### CartPole PPO

```text
train_cartpole.py
```

Trains a PPO agent on CartPole-v1.

```text
eval_cartpole.py
```

Evaluates the trained PPO CartPole agent over 10 episodes.

---

### CartPole DQN

```text
train_cartpole_dqn.py
```

Trains a DQN agent on CartPole-v1.

```text
eval_cartpole_dqn.py
```

Evaluates the trained DQN CartPole agent over 10 episodes.

---

### Atari Breakout PPO

```text
train_atari.py
```

Trains a PPO agent on Atari Breakout-v5 using CNN policy, Atari preprocessing, and frame stacking.

```text
eval_atari.py
```

Evaluates the trained PPO Atari Breakout agent over 10 episodes.

---

### Atari Breakout DQN

```text
train_atari_dqn.py
```

Trains a DQN agent on Atari Breakout-v5 using CNN policy.

```text
eval_atari_dqn.py
```

Evaluates the trained DQN Atari Breakout agent over 10 episodes.

---

### Random Agent Baseline

```text
eval_random_atari.py
```

Evaluates a random agent on Atari Breakout-v5 as a baseline.

---

### Environment Test

```text
test_atari.py
```

Checks whether the Atari Breakout-v5 environment can be created successfully.

---

### Plotting Scripts

```text
plot_results.py
```

Saves main experiment results and generates basic reward comparison plots.

```text
plot_learning_rate.py
```

Generates the learning-rate comparison figure for PPO on CartPole-v1.

```text
plot_final_figures.py
```

Generates the final figures used in the report.

---

## How to Run

### 1. Train PPO on CartPole-v1

```bash
python train_cartpole.py
```

Evaluate the model:

```bash
python eval_cartpole.py
```

---

### 2. Train DQN on CartPole-v1

```bash
python train_cartpole_dqn.py
```

Evaluate the model:

```bash
python eval_cartpole_dqn.py
```

---

### 3. Test Atari Environment

```bash
python test_atari.py
```

---

### 4. Train PPO on Atari Breakout-v5

```bash
python train_atari.py
```

Evaluate the model:

```bash
python eval_atari.py
```

---

### 5. Train DQN on Atari Breakout-v5

```bash
python train_atari_dqn.py
```

Evaluate the model:

```bash
python eval_atari_dqn.py
```

---

### 6. Evaluate Random Agent on Atari Breakout-v5

```bash
python eval_random_atari.py
```

---

### 7. Generate Figures

```bash
python plot_final_figures.py
```

The generated figures will be saved in the `results/` folder.

---

## Experiments

### Experiment 1: Training Steps on CartPole-v1

| Environment | Algorithm | Training Steps | Average Reward |    Std |
| ----------- | --------- | -------------: | -------------: | -----: |
| CartPole-v1 | PPO       |         10,000 |          292.2 | 104.02 |
| CartPole-v1 | PPO       |         50,000 |          500.0 |    0.0 |

---

### Experiment 2: Learning Rate on CartPole-v1

| Environment | Algorithm | Training Steps | Learning Rate | Average Reward |    Std |
| ----------- | --------- | -------------: | ------------: | -------------: | -----: |
| CartPole-v1 | PPO       |         10,000 |        0.0001 |          244.4 |  61.10 |
| CartPole-v1 | PPO       |         10,000 |        0.0003 |          292.2 | 104.02 |
| CartPole-v1 | PPO       |         10,000 |         0.001 |          491.7 |  24.90 |

---

### Experiment 3: Training Steps on Atari Breakout-v5

| Environment | Algorithm | Training Steps | Average Reward | Std |
| ----------- | --------- | -------------: | -------------: | --: |
| Breakout-v5 | PPO       |         50,000 |            0.4 | 1.2 |
| Breakout-v5 | PPO       |        100,000 |            0.4 | 0.8 |
| Breakout-v5 | PPO       |        200,000 |            0.4 | 1.2 |

---

### Experiment 4: Learning Rate on Atari Breakout-v5

| Environment | Algorithm | Training Steps | Learning Rate | Average Reward |   Std |
| ----------- | --------- | -------------: | ------------: | -------------: | ----: |
| Breakout-v5 | PPO       |        100,000 |        0.0001 |            0.5 | 0.806 |
| Breakout-v5 | PPO       |        100,000 |       0.00025 |            0.4 |   0.8 |
| Breakout-v5 | PPO       |        100,000 |        0.0005 |            1.4 |  2.20 |

---

### Experiment 5: Random Agent, DQN, and PPO on Atari Breakout-v5

| Environment | Agent / Algorithm | Training Steps | Learning Rate | Average Reward |  Std |
| ----------- | ----------------- | -------------: | ------------: | -------------: | ---: |
| Breakout-v5 | Random Agent      |              0 |             - |            0.3 | 0.46 |
| Breakout-v5 | DQN               |        100,000 |        0.0001 |            0.9 | 1.58 |
| Breakout-v5 | PPO               |        100,000 |        0.0005 |            1.4 | 2.20 |

---

### Experiment 6: PPO vs DQN on CartPole-v1

| Environment | Algorithm | Training Steps | Average Reward |    Std |
| ----------- | --------- | -------------: | -------------: | -----: |
| CartPole-v1 | PPO       |         10,000 |          292.2 | 104.02 |
| CartPole-v1 | DQN       |         10,000 |          185.6 |  56.50 |
| CartPole-v1 | PPO       |         50,000 |          500.0 |    0.0 |
| CartPole-v1 | DQN       |         50,000 |           90.5 |  46.85 |

---

## Summary

The results show that PPO can solve CartPole-v1 efficiently.
With 50,000 training steps, PPO achieved the maximum reward of 500.

Atari Breakout-v5 was much more difficult.
Even when the training steps increased to 200,000, the PPO agent did not learn a stable Breakout strategy.

The main reasons are:

* CartPole-v1 uses low-dimensional numerical observations.
* Atari Breakout-v5 uses image observations.
* CartPole-v1 provides dense rewards.
* Atari Breakout-v5 has sparse rewards.
* Atari training requires more computation and longer training time.

Overall, the experiments show that training steps, learning rate, and algorithm choice all affect reinforcement learning performance.

---

## References

1. Gymnasium Documentation: https://gymnasium.farama.org/
2. Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
3. Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
4. Mnih et al., "Human-level control through deep reinforcement learning", Nature, 2015.
5. Gymnasium Atari Environments: https://gymnasium.farama.org/environments/atari/
