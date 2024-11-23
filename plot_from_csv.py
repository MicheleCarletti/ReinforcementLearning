import csv
import os
import matplotlib.pyplot as plt

session_name = "tr_DQN_256h_6000e_22-11-2024_PER_hpc"
file_name = os.path.join("data/results/",session_name)
file_name = file_name + ".csv"

reward_history = []
ws = 100

def moving_average(data, window_size):
    """ Compute the moving average of window_size elements"""
    if window_size <= 0:
        raise ValueError("Window size <= 0")
    mavg = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        mavg.append(sum(window) / window_size)
    return mavg

with open(file_name, mode='r') as f:
    csvreader = csv.DictReader(f, delimiter=';')

    for r in csvreader:
        reward_history.append(float(r['Total Reward']))


moving_avg = moving_average(reward_history, ws)

plt.figure(figsize=(10, 8))
plt.plot(reward_history, label='Reward per episode')
plt.plot(moving_avg, label=f'Moving average ({ws} episodes)')
plt.title(f"Total Rewards\n[{session_name}]")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid()
plt.show()
