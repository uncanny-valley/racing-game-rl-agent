import math

def compute_reward(telemetry):
    normalized_speed = telemetry.speed / telemetry.max_speed

    if math.isclose(normalized_speed, 0.):
        return -1.
    elif abs(telemetry.position_x) > 0.8:
        return -0.8
    else:
        return normalized_speed

def sum_rewards(telemetries):
    return sum([compute_reward(t) for t in telemetries])