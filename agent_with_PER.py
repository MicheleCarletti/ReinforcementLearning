import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
import datetime
import time
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
        self.h2 = num_hidden // 2
        self.fc1 = nn.Linear(state_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, self.h2)
        self.fc3 = nn.Linear(self.h2, action_dim)

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


def train(memory, policy_net, target_net, optimizer, batch_size, gamma, beta, losses):
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
    loss = (weights * (q_values.squeeze() - target_q_values)**2).mean()  # Weighted MSE
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)  # gradient clipping
    optimizer.step()

    losses.append(loss.item())
    # Update PER priorities
    priorities = (q_values.squeeze() - target_q_values).abs().cpu().detach().numpy() + 1e-5
    memory.update_priorities(indicies, priorities)

def prepare_results(reward_history, mar, loss_history, hpc):
    """ Analize training results"""

    if not hpc:

        # Plotting rewards
        plt.figure(figsize=(10, 8))
        plt.plot(reward_history, label='Reward per episode')
        plt.plot(mar, label='Moving average (100 episodes)')
        plt.title("Total Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid()
        plt.show()

        # Plotting loss
        plt.figure(figsize=(10, 8))
        plt.plot(loss_history)
        plt.title("Loss per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid()
        plt.show()

if __name__ == "__main__":

    #### Model parameters ###
    hidden_units = 128 # Number of hidden neurons
    gamma = 0.99    # Discount factor
    epsilon = 1.0   # Initial exploration probability
    epsilon_min = 0.01  
    epsilon_decay = 0.995
    learning_rate = 0.0005
    batch_size = 64
    max_memory_size = 100000
    n_episodes = 1600
    target_net_freq = 10    # Update frequency for target network
    alpha = 0.6
    beta = 0.4
    beta_increment_per_episode = 0.001
    running_on_hpc = False
    #######################

    # Set-up the environment
    env = gym.make("LunarLander-v3", render_mode="None")
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
    loss_history = []
    model_saved = False

    # Rewards moving average
    moving_avg_period = 100
    moving_avg_rewards = []

    session_name = f"DQN_{hidden_units}h_{n_episodes}e_{datetime.datetime.now().strftime('%d-%m-%Y')}_PER"

    with open(f"data/results/tr_{session_name}.csv", mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Episode", "Total Reward", "Epsilon", "Elapsed time [s]"]) # Header

        # Start the training 
        for episode in range(n_episodes):
            start_time = time.time()
            state, _ = env.reset(seed=random.randint(0, 1000))
            done = False
            total_reward = 0
            episode_loss = []
            n_step = 0


            while not done:
                action = select_action(state, policy_net, epsilon, action_dim)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                memory.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward


                train(memory, policy_net, target_net, optimizer, batch_size, gamma, beta, episode_loss)
                n_step += 1
        
                
            rewards_history.append(total_reward)    # Track rewards

            # Compute moving average
            if len(rewards_history) >= moving_avg_period:
                moving_avg_rewards.append(np.mean(rewards_history[-moving_avg_period:]))
            else:
                moving_avg_rewards.append(np.mean(rewards_history))

            episode_loss = np.array(episode_loss)
            loss_history.append(episode_loss.sum() / len(episode_loss))

            # Early stop condition to avoid overfitting
            if total_reward > 5000:
                print(f"Training stopped as episode {episode + 1} achieved a good total reward: {total_reward:.2f}")
                torch.save(policy_net.state_dict(), f"models/models_with_PER/{session_name}.pth") # Save the model
                model_saved = True
                break
            
            # Decrease the epsion value
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
        
            beta = min(1.0, beta + beta_increment_per_episode)  # Update beta value

            # Update the target network
            if episode % target_net_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}, ET: {elapsed_time:.2f} s")
            # Write episode data to CSV
            writer.writerow([episode+1, round(total_reward,3), round(epsilon,3), round(elapsed_time,2)])

    # If the training is completed save the model
    if not model_saved:
        torch.save(policy_net.state_dict(), f"models/models_with_PER/{session_name}.pth") # Save the model

    env.close()

    prepare_results(rewards_history, moving_avg_rewards, loss_history, running_on_hpc)


