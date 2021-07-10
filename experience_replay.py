import collections
import numpy as np
import random
from transition import Transition

class ExperienceReplay:
    def __init__(self, size: int):
        self.max_size = size
        self._memory = collections.deque(maxlen=size)

    def __len__(self):
        return len(self._memory)

    def add_transition(self, state, action, reward, next_state, is_terminal, is_initial_state):
        t = Transition(state, action, reward, next_state, is_terminal, is_initial_state)
        self._memory.append(t)

    def sample_minibatch(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=False)
        states, actions, rewards, next_states, terminals, is_initial_states = zip(*[self._memory[idx] for idx in indices])
        return np.array(states).reshape(batch_size, 84, 84, 3), np.array(actions), np.array(rewards, dtype=np.float32), \
                 np.array(terminals, dtype=np.uint8), np.array(next_states).reshape(batch_size, 84, 84, 3), \
                 np.array(is_initial_states, dtype=np.uint8)