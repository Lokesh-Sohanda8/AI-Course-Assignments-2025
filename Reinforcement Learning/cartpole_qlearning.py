# cartpole_qlearning.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

def moving_avg(x, w=50):
    if len(x) < w: return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode="valid")

# Output directory
outdir = "rl_outputs"
os.makedirs(outdir, exist_ok=True)

env = gym.make("CartPole-v1")
n_actions = env.action_space.n

# MUCH better discretization (finer grid)
position_bins   = np.linspace(-4.8, 4.8, 20)
velocity_bins   = np.linspace(-3.0, 3.0, 20)
angle_bins      = np.linspace(-0.42, 0.42, 40)
ang_vel_bins    = np.linspace(-3.0, 3.0, 40)

bins = [position_bins, velocity_bins, angle_bins, ang_vel_bins]

def discretize(obs):
    return tuple(int(np.digitize(obs[i], bins[i])) for i in range(len(obs)))

# Bigger Q-table → more expressive
Q = np.zeros((21,21,41,41,n_actions))

# Hyperparameters (optimized)
alpha = 0.15
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.005
epsilon_decay = 0.997  # slower decay = more exploration

episodes = 3000
max_steps = 500

rewards = []

for ep in range(1, episodes+1):

    obs, info = env.reset()
    state = discretize(obs)
    total_reward = 0

    for step in range(max_steps):

        # ε-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_obs)

        # Reward shaping → greatly speeds up learning
        shaped_reward = reward + (abs(next_obs[2]) * -2.0)

        # Q Update
        Q[state + (action,)] += alpha * (
            shaped_reward + gamma * np.max(Q[next_state]) - Q[state + (action,)]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if ep % 200 == 0:
        print(f"[CartPole Optimized] Ep {ep}/{episodes}  Avg(last200) = {np.mean(rewards[-200:]):.2f}")

# Save results
np.save(os.path.join(outdir, "Q_cartpole_optimized.npy"), Q)

# Plot
plt.plot(rewards)
ma = moving_avg(rewards, 50)
plt.plot(range(len(ma)), ma)
plt.title("CartPole Optimized Q-learning")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig(os.path.join(outdir, "cartpole_optimized.png"))
plt.show()
