# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from replay_buffer import ReplayBuffer


def act(policy_net, state, action_size, epsilon):
    """Select an action based on epsilon-greedy policy."""
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action_values = policy_net(state)
    return torch.argmax(action_values, dim=1).item()


def replay(memory, batch_size, policy_net, target_net, optimizer, criterion, gamma):
    """Train the network with samples from the replay buffer."""
    if len(memory) < batch_size:
        return

    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.FloatTensor(dones).unsqueeze(1)

    current_q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = criterion(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def update_target_network(policy_net, target_net):
    """Update the target network to match the policy network."""
    target_net.load_state_dict(policy_net.state_dict())


def train_dqn(env, policy_net, target_net, n_episodes, batch_size, learning_rate, gamma, epsilon, epsilon_min,
              epsilon_decay, memory_size):
    """Train the DQN with the specified hyperparameters."""
    memory = ReplayBuffer(memory_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scores = []

    for e in range(n_episodes):
        state, _ = env.reset()
        score = 0
        for time_step in range(500):
            if e % 100 == 0:
                env.render()
            action = act(policy_net, state, env.action_space.n, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = reward if not done else -10
            memory.remember(state, action, reward, next_state, done)
            state = next_state
            score += 1
            if done:
                scores.append(score)
                break
            replay(memory, batch_size, policy_net, target_net, optimizer, criterion, gamma)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        update_target_network(policy_net, target_net)

        if e % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode: {e}/{n_episodes}, Average Score: {avg_score}, Epsilon: {epsilon:.2}")

    env.close()
    print("Training finished.\n")
