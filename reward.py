import math

def compute_reward(telemetry):
    normalized_speed = telemetry.speed / telemetry.max_speed
    magnitude_x = abs(telemetry.position_x)

    if math.isclose(normalized_speed, 0.):
        return -1.
    elif magnitude_x > 0.8:
        return -10. * magnitude_x * normalized_speed
    else:
        multiplier = 5. if magnitude_x < 0.2 else (-5 * magnitude_x)
        return  multiplier * normalized_speed

def sum_rewards(telemetries):
    return sum([compute_reward(t) for t in telemetries])