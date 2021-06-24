import numpy as np


class Telemetry:
    def __init__(self, position_x: np.float64, speed: np.float64, max_speed: np.float64):
        self._position_x = position_x
        self._speed = speed
        self._max_speed = max_speed

    @property
    def position_x(self):
        return self._position_x

    @property
    def speed(self):
        return self._speed

    @property
    def max_speed(self):
        return self._max_speed

    @classmethod
    def from_json(cls, blob):
        position_x = blob.get('position_x')
        speed = blob.get('speed')
        max_speed = blob.get('max_speed')
        return Telemetry(position_x, speed, max_speed)