from collections import namedtuple

Transition = namedtuple('Transition', field_names=[
    'state',
    'action',
    'reward',
    'next_state',
    'is_terminal',
    'is_initial_state'])
