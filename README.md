# DQN for lunar landing v3

This project presents a RL agent trying to solve the OpenAI Gymnasium LunarLanding-v3 environment.

**Structure**

`data` reward plots. Format **hu_#HiddenUnits_#Epoches_#EasrlyStopped.png**

`models` pretarined models in format **DQN_lunar_lander#HiddenUnitsh_#episodese_date.pth**

`agent.py` Implement a DQN based agent with Replay Buffer

`agent_with_PER.py` Implement a DQN based agent with Prioritized Experience Replay (PER) Buffer


## DQN-based agent with Replay Buffer
Classic DQN approach. A policy network tries to approximate the Q Function. It performs an _epsilon-greedy_ policy.

**Parameters**

* hidden_units: number of hidden neuron per layer in the DQN
* batch_size: number of experience per batch (typical value 64)
* learning rate: optimizer's step magnitude (typical value 0.001)
* n_episodes: number of epochs in the training loop
* gamma: discount factor in target q values computation (typical value 0.99)
* epsilon: initial exploration probability (typical value 1)
* epsilon_min: minimum value for epsilon (typical value 0.01)
* epsilon_decay: decay factor for epsilon. Close to 1 -> slow decay, close to 0 -> fast decay. Epsilon value is updated after each episode
* max_memory_size: maximu number of experiences in the replay buffer
* target_net_frequency: number of episodes between each target network update


## DQN-based agent with PER Buffer

DQN approach with Prioritized Experience Replay (PER) Buffer. Examples used for training are sampled accordin to a probability distribution derived from the priority of each experience.

**Parameter**

Same of the previous agent, plus
* alpha: prioritization exponent. Controls how the buffer will encourage the importat experiences (the ones with higer error). Close to 1 -> higer prority experiences are more likely to be selected. Close to 0 -> more random sampling. Alpha = 0 -> Class replay buffer (typical value 0.6)
* beta: importance sampling correction coefficient. It tries to balance the bias introduced by alpha. It determines the weight during the gradients update process. At the beginnig it should be low, so the model learns more from important experience, while the training proceed, beta value is increade in order to compensate the learing bias (typical initial value 0.4)
* beta_increment_per_episode: after each episode beta is increased to compensate the learning bias induced by alpha (typical value 0.001)

## Testing
File `test.py` contains a simple test for pretrained models.

File `test_batch.py` allows to test multiple models under the same conditions. Models parameters `.pth` files are in `test_batch` folder.

Currently the best performing version in `DQN_256h_10000e_23-11-2024_PER_hpc.pth` with a success probability of 1.0 on 80000 test episodes.

# Author

|Name|Email|
|----|-----|
|Michele Carletti|michelecarletti98@gmail.com|




