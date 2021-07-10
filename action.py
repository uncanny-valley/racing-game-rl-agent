from bidict import bidict
from enum import IntEnum, unique
import numpy as np


@unique
class Action(IntEnum):
    ACCELERATE = 0
    ACCELERATE_LEFT = 1
    ACCELERATE_RIGHT = 2

    # (down, up, left, right) => action name
    __action_keys_bijective_map__ = bidict({
        (0, 1, 0, 0): ACCELERATE,
        (0, 1, 1, 0): ACCELERATE_LEFT,
        (0, 1, 0, 1): ACCELERATE_RIGHT
      })

    @classmethod
    def __keys_to_action(cls, keys):
        """
        :param cls (Action): class
        :param keys (Tuple<int>): A quadruple of binary values corresponding to (down, up, left, right)
                                  where a set bit indicates that the key was pressed
        :returns: the corresponding action taken
        :rtype: Action
        :raises: ValueError if an invalid key configuration was pressed
        """
        if keys in cls.__action_keys_bijective_map__:
            return cls(cls.__action_keys_bijective_map__[keys])
        else:
            raise ValueError(f'Given set of keys is an invalid configuration: {keys}')

    @classmethod
    def num_actions(cls):
        return len(cls.__members__)

    @classmethod
    def random_action(cls):
        return cls(np.random.randint(0, cls.num_actions()))

    @classmethod
    def from_array(cls, keys):
        if len(keys) != 4:
            raise ValueError(f'Size of `keys` must correspond to the number of possible keys (down, up, left, right). Expected 4, but given {len(keys)}')
        
        return cls.from_keys(keys[0], keys[1], keys[2], keys[3])

    @classmethod
    def from_keys(cls, down, up, left, right):
        return cls.__keys_to_action((down, up, left, right))

    def to_keys(self):
        return self.__action_keys_bijective_map__.inverse[self.value]