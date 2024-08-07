# replay_buffer.py
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
