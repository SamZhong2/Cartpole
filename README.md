
# DQN CartPole

This repository contains an implementation of Deep Q-Learning (DQN) to solve the CartPole-v1 environment from OpenAI Gym.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Hyperparameters](#hyperparameters)
- [Results](#results)


## Introduction

Deep Q-Learning (DQN) is a reinforcement learning algorithm that combines Q-Learning with deep neural networks to solve problems with high-dimensional state spaces. This project demonstrates the use of DQN to balance a pole on a cart in the CartPole-v1 environment.

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/your-repository.git
    cd your-repository
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scriptsctivate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the DQN model, run the following command:

```sh
python main.py
```

### Running a Pre-Trained Model

To run a pre-trained model and see it in action, run:

```sh
python main.py run
```

## Project Structure

```
.
├── dqn.py               # DQN model definition
├── main.py              # Main script to train or run the model
├── replay_buffer.py     # Replay buffer implementation
├── run.py               # Script to run the pre-trained model
├── train.py             # Training loop and related functions
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

## Hyperparameters

The main hyperparameters used in this project are:

- `state_size`: 4 (size of the state space)
- `action_size`: 2 (number of actions)
- `batch_size`: 64
- `n_episodes`: 1000 (number of episodes for training)
- `learning_rate`: 0.0005
- `gamma`: 0.99 (discount factor)
- `epsilon`: 1.0 (initial exploration rate)
- `epsilon_min`: 0.01 (minimum exploration rate)
- `epsilon_decay`: 0.995
- `memory_size`: 1000000 (replay buffer size)

## Results

During training, the model's performance improves over time. The average score per 100 episodes is printed to the console.

Example output:

```
Episode: 100/1000, Average Score: 10.5, Epsilon: 0.90
Episode: 200/1000, Average Score: 25.3, Epsilon: 0.82
...
```

