import gymnasium as gym
import torch
from agent import DQN
import numpy as np

# Environment set-up
env = gym.make("LunarLander-v3", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_units = 128

# Define the pre-trained model
model = DQN(state_dim, action_dim, hidden_units).to(device)
model.load_state_dict(torch.load(f"models/models_with_PER/DQN_{hidden_units}h_1700e_22-11-2024_PER.pth", map_location=torch.device(device)))
model.eval()

def select_action(state, policy_net):
    """ Choose the action with deterministic policy"""
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state)
    return torch.argmax(q_values).item()    # Choose the action with higher Q value


# Play for 10 episodes
reward_res = []
epoches = 50
for episode in range(epoches):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = select_action(state, model)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    
    print(f"Episode {episode + 1}/{epoches}, Total reward: {total_reward:.2f}")

    reward_res.append(total_reward)

succ_prob = np.array([1.0 for x in reward_res if x >= 200]).sum() / epoches
print(f"\nSuccess probability: {succ_prob:.2f}")

env.close()
