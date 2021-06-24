from collections import namedtuple

Transition = namedtuple('Transition', field_names=[
    'current_state',
    'action',
    'reward',
    'is_terminal',
    'next_state',
    'is_initial_state'])
