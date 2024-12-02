"""
@author Michele Carletti
Test a bunch of pre-trained models under the same conditions
"""
import os
import re
from agent import DQN
import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

num_epochs = 200
folder_path = "./test_batch/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_seed = np.random.randint(1,50)
running_on_hpc = False

def extract_hu(filename):
    """ Given model's name, returns the number of hidden units in first layer (hu)"""
    # Regex to extract hu number
    match = re.search(r'_(\d+)h', filename)
    if match:
        return int(match.group(1))  # From string to int
    return None

def select_action(state, policy_net):
    """ Choose the action with deterministic policy"""
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state)
    return torch.argmax(q_values).item()    # Choose the action with higher Q value

def moving_average(data, window_size):
    """ Compute the moving average of window_size elements"""
    if window_size <= 0:
        raise ValueError("Window size <= 0")
    mavg = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        mavg.append(sum(window) / window_size)
    return mavg

def run_test(fpath, epoches, state_dim, action_dim, env):
    """ Run a complete test session. Each model is tested under the same conditions"""
    models_name = []
    models = []

    # Load models from folder
    for f in os.listdir(fpath):
        if os.path.isfile(os.path.join(fpath, f)):  # Is the element a file?
            hu = extract_hu(f)
            if hu is None:
                raise ValueError("Not able to get the hu value!")
            
            models_name.append(f)

            model = DQN(state_dim, action_dim, hu)
            model.load_state_dict(torch.load(os.path.join(fpath, f), map_location=torch.device(device)))
            model.eval()

            models.append(model)    # Save the model in a list
    
    # Start test
    model_rewards = np.zeros((len(models), epoches))    
    for e in tqdm(range(epoches), desc="Testing", unit="episode"):    # Iterate on episodes


        for i in range(len(models)):    # Iterate on models

            seed = start_seed + e   # Guarantee that each model is tested in the same environment
            state,_ = env.reset(seed=seed)
            done = False
            total_reward = 0
            


            while not done:
                action = select_action(state, models[i])
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward
                
            model_rewards[i,e] = total_reward  # Rewards of the model under test

    # Compute success probability per model
    succ_prob = []
    for i in range(len(model_rewards)):
        succ_prob.append(np.sum(model_rewards[i] >= 200) / model_rewards.shape[1])
    
    return model_rewards, models_name, succ_prob

if __name__ == "__main__":

    environment = gym.make("LunarLander-v3", render_mode="None")
    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.n

    results, names, prob = run_test(folder_path, num_epochs, state_dim, action_dim, environment)

    #print(results.shape)

    for name, p in zip(names, prob):
        # Splits name_file.ext in (name_file, .ext)
        print(f"Model: {os.path.splitext(name)[0]}   success probability: {p:.2f}")

    plt.figure(figsize=(10,8))
    
    # Plot results
    for i in range(len(results)):
        lb = os.path.splitext(names[i])[0]

        # If there are too many episodes plot moving averages on 100 episodes
        if num_epochs > 100:
            moving_avg = moving_average(results[i], 100)
            plt.plot(moving_avg, label=lb)
            plt.title(f"Test batch: moving average (100 episodes)\n{num_epochs} epoches")
        else:
            plt.plot(results[i], label=lb)
            plt.title(f"Test batch: total rewards per episode\n{num_epochs} epoches")

    
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()

    if running_on_hpc:
        plt.savefig(f'./data/Test_batch_{num_epochs}e.png')
    else:
        plt.show()

                



