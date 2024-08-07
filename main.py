# main.py
import gym
import torch
import sys
from dqn import DQN
from train import train_dqn
from run import load_and_run_model


def main():
    env = gym.make('CartPole-v1')

    # Hyperparameters
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    batch_size = 64
    n_episodes = 1000
    learning_rate = 0.001
    gamma = 0.99  # discount rate
    epsilon = 1.0  # exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995
    memory_size = 1000000

    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        # Load and run a pre-trained model
        model_path = 'dqn_cartpole.pth'
        load_and_run_model(env, model_path, episodes=5)
    else:
        # Initialize the DQN
        policy_net = DQN(state_size, action_size)
        target_net = DQN(state_size, action_size)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        # Train the DQN
        train_dqn(env, policy_net, target_net, n_episodes, batch_size, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay, memory_size)

        # Save the trained model
        torch.save(policy_net.state_dict(), 'dqn_cartpole.pth')
        print("Model saved.")


if __name__ == "__main__":
    main()
