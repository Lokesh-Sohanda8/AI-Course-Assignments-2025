import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

def moving_avg(x, w=100):
    if len(x) < w: return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='valid')

# Output folder
outdir = "rl_outputs"
os.makedirs(outdir, exist_ok=True)

# Create environment
env = gym.make("MountainCar-v0")
n_actions = env.action_space.n

# ---- Correct State Ranges ----
# Empirically correct ranges used by the actual environment
pos_min, pos_max = -1.2, 0.6
vel_min, vel_max = -0.07, 0.07

# ---- Good Binning Strategy ----
# 40–50 bins is enough. More will make Q-table too large.
n_bins = 40
pos_bins = np.linspace(pos_min, pos_max, n_bins)
vel_bins = np.linspace(vel_min, vel_max, n_bins)

def discretize(obs):
    pos, vel = obs
    pos_bin = np.digitize(pos, pos_bins)
    vel_bin = np.digitize(vel, vel_bins)
    return (pos_bin, vel_bin)

# Q table (41x41x3)
Q = np.zeros((n_bins+1, n_bins+1, n_actions))

# ---- Stable Hyperparameters ----
alpha = 0.1          # smaller = more stable
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.02
epsilon_decay = 0.999

episodes = 8000
max_steps = 200

rewards = []

for ep in range(1, episodes + 1):

    obs, info = env.reset()
    state = discretize(obs)
    total_reward = 0

    for step in range(max_steps):

        # ε-greedy exploration
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_obs)

        # ---- Gentle Reward Shaping (proven best practice) ----
        shaped = r
        shaped += abs(next_obs[1]) * 0.2         # encourage speed
        shaped += (next_obs[0] + 1.2) * 0.3      # reward moving right

        # ---- Q-learning update ----
        td_target = shaped + gamma * np.max(Q[next_state])
        Q[state + (action,)] += alpha * (td_target - Q[state + (action,)])

        state = next_state
        total_reward += r

        if done:
            break

    rewards.append(total_reward)

    # ---- Epsilon Annealing ----
    if ep % 2000 == 0:        # warm restart
        epsilon = 1.0
    else:
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # ---- Console Progress ----
    if ep % 500 == 0:
        avg500 = np.mean(rewards[-500:])
        print(f"[Improved MountainCar] Episode {ep}/{episodes} | Avg500 = {avg500:.2f} | ε={epsilon:.3f}")

# ------------------ Save Results ------------------
np.save(os.path.join(outdir, "Q_mountain_improved.npy"), Q)

# Plotting
plt.plot(rewards)
ma = moving_avg(rewards, 100)
plt.plot(range(len(ma)), ma)
plt.title("MountainCar Improved Tabular Q-learning")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig(os.path.join(outdir, "mountaincar_improved.png"))
plt.show()
