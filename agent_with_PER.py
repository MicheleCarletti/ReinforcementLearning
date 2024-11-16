import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt


class PER:
    def __init__(self, capacity, alpha=0.6):
        """ Prioritized Experience Replay (PER)\n Buffer with priority"""
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha
    
    def add(self, state, action, reward, next_state, done):
        """ Add an experience to the buffer with the corresponding priority value"""
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """ Sample batch_size elements from the buffer.\n Probabilities are derived from priorities"""
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:self.position])
        scaled_priorities = priorities ** self.alpha
        probabilities = scaled_priorities / scaled_priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        return (states, actions, rewards, next_states, dones), indices, torch.FloatTensor(weights)
    
    def update_priorities(self, indicies, priorities):
        """ Update experiences priorities"""
        for idx, priority in zip(indicies, priorities):
            self.priorities[idx] = priority

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


def train(memory, policy_net, target_net, optimizer, batch_size, gamma, beta):
    """ Train the DQN model"""
    if len(memory.buffer) < batch_size:
        return
    
    (states, actions, rewards, next_states, dones), indicies, weights = memory.sample(batch_size, beta)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)
    weights = weights.to(device)

    # Compute actual Q values
    q_values = policy_net(states).gather(1, actions)

    # Compute target Q values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Compute the loss
    loss = (torch.FloatTensor(weights) * (q_values.squeeze() - target_q_values)**2).mean()  # Weighted MSE
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update PER priorities
    priorities = (q_values.squeeze() - target_q_values).abs().cpu().detach().numpy() + 1e-5
    memory.update_priorities(indicies, priorities)

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
    alpha = 0.6
    beta = 0.4
    beta_increment_per_episode = 0.001

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
    memory = PER(max_memory_size, alpha)

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
            
            memory.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward


            train(memory, policy_net, target_net, optimizer, batch_size, gamma, beta)

            # Check if total reward is so bad
            if total_reward < -150:
                print(f"Episode {episode + 1} ended early due to low reward!")
                break
                
        rewards_history.append(total_reward)    # Track rewards

        # Check for early stopping for the overall training
        if total_reward > 270:
            print(f"Training stopped as episode {episode + 1} achieved a good total reward: {total_reward:.2f}")
            torch.save(policy_net.state_dict(), f"models/models_with_PER/DQN_lunar_lander_{hidden_units}.pth") # Save the model
            model_saved = True
            break
            
        # Decrease the epsion value
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        beta = min(1.0, beta + beta_increment_per_episode)  # Upfdate beta value

        # Update the target network
        if episode % target_net_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward:.2f}")

    # If the training is completed save the model
    if not model_saved:
        torch.save(policy_net.state_dict(), f"models/models_with_PER/DQN_lunar_lander_{hidden_units}.pth") # Save the model

    env.close()

    # Plotting rewards
    fig = plt.figure(figsize=(10,8))
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Total Reward per episode")
    plt.grid()
    plt.show()

