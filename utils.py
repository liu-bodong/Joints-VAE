
import torch
import numpy as np

def annealing(step, k=0.0025, x0=2500):
    """
    Sigmoid annealing function.
    Args:
        step (int): Current training step.
        k (float): Steepness of the curve.
        x0 (int): Midpoint of the curve.
    """
    return float(1 / (1 + torch.exp(-k * (step - x0))))


def ramp(step, rampup_length=1000, direction=1):
    """
    Exponential ramp function.
    Args:
        step (int): Current training step.
        rampup_length (int): Length of the ramp-up period.
        direction (int): 1 for ramp-up, -1 for ramp-down.
    Returns:
        float: Ramp value between 0 and 1.
    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(step, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        if direction == 1:
            return float(np.exp(-5.0 * phase * phase))
        else:
            return float(1.0 - np.exp(-5.0 * phase * phase))