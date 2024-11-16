import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt



# Neural network for Q function
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_hidden):
        """ Define the model to learn the Q function"""
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def select_action(state, policy_net, epsilon, action_dim):
    """ Choose the action via epsilon-greedy policy"""
    if np.random.rand() < epsilon:
        return random.randrange(action_dim)  # Explore: choose a random action
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state)
    return torch.argmax(q_values).item()  # Exploit: choose the action with the maximum Q value


def train(memory, policy_net, target_net, optimizer, batch_size, gamma):
    """ Train the DQN model"""
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Compute actula Q values
    q_values = policy_net(states).gather(1, actions)

    # Compute target Q values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Compute the loss
    loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":

    # Model parameters
    hidden_units = 1024 # Number of hidden neurons
    gamma = 0.99    # Discount factor
    epsilon = 1.0   # Initial exploration probability
    epsilon_min = 0.01  
    epsilon_decay = 0.995
    learning_rate = 0.001
    batch_size = 64
    max_memory_size = 10000
    n_episodes = 800
    target_net_freq = 15    # Update frequency for target network

    # Set-up the environment
    env = gym.make("LunarLander-v3", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Define policy and target networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    policy_net = DQN(state_dim, action_dim, hidden_units).to(device)
    target_net = DQN(state_dim, action_dim, hidden_units).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = deque(maxlen=max_memory_size)

    # Save rewards
    rewards_history = []
    model_saved = False
    # Start the training 
    for episode in range(n_episodes):
        state, _ = env.reset(seed=random.randint(0, 1000))
        done = False
        total_reward = 0


        while not done:
            action = select_action(state, policy_net, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward


            train(memory, policy_net, target_net, optimizer, batch_size, gamma)

            # Check if total reward is so bad
            if total_reward < -150:
                print(f"Episode {episode + 1} ended early due to low reward!")
                break
                
        rewards_history.append(total_reward)    # Track rewards

        # Check for early stopping for the overall training
        if total_reward > 270:
            print(f"Training stopped as episode {episode + 1} achieved a good total reward: {total_reward:.2f}")
            torch.save(policy_net.state_dict(), f"models/DQN_lunar_lander_{hidden_units}.pth") # Save the model
            model_saved = True
            break
            
        # Decrease the epsion value
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Update the target network
        if episode % target_net_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward:.2f}")

    # If the training is completed save the model
    if not model_saved:
        torch.save(policy_net.state_dict(), f"models/DQN_lunar_lander_{hidden_units}.pth") # Save the model

    env.close()

    # Plotting rewards
    fig = plt.figure(figsize=(10,8))
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Total Reward per episode")
    plt.grid()
    plt.show()

