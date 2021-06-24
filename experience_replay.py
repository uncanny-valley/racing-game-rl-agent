import collections
import numpy as np
import random
from transition import Transition

class ExperienceReplay:
    def __init__(self, size: int):
        self.max_size = size
        self._memory = collections.deque(maxlen=size)

    @property
    def size(self):
        return len(self._memory)

    def add_transition(self, state, action, reward, next_state, is_terminal, is_initial_state):
        t = Transition(state, action, reward, next_state, is_terminal, is_initial_state)
        self._memory.append(t)

    def sample_minibatch(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        states, actions, rewards, terminals, next_states = zip([self._memory[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(terminals, dtype=np.uint8), np.array(next_states)