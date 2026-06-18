"""
Event encoding: bridge between continuous Gymnasium state and discrete spike inputs.

CartPole-v1 state: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
Each dimension is encoded as a 2-bit ON/OFF spike pair, yielding an 8-dim spike vector.
"""

import torch
import numpy as np
from typing import Optional


DEFAULT_THRESHOLDS = [0.05, 0.1, 0.02, 0.1]


def velocity_to_spikes(v: float, threshold: float = 0.1) -> list[int]:
    """
    Convert a continuous scalar to a 2-bit spike event.

    ON event  [1, 0]: value exceeded positive threshold
    OFF event [0, 1]: value exceeded negative threshold
    No event  [0, 0]: value in dead zone
    """
    if v > threshold:
        return [1, 0]
    elif v < -threshold:
        return [0, 1]
    else:
        return [0, 0]


def state_to_spikes(
    obs: np.ndarray,
    thresholds: Optional[list[float]] = None,
) -> list[int]:
    """
    Full CartPole state [pos, vel, angle, angular_vel] -> 8-dim spike vector.
    Applies velocity_to_spikes independently to each state dimension.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    spikes = []
    for val, thr in zip(obs, thresholds):
        spikes.extend(velocity_to_spikes(float(val), thr))
    return spikes


def encode_batch(
    obs_batch: torch.Tensor,
    thresholds: Optional[list[float]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Vectorised event encoding for a batch of observations.

    Args:
        obs_batch: (batch, 4) tensor of CartPole states.
        thresholds: per-dimension thresholds; defaults to DEFAULT_THRESHOLDS.
        device: target device for the output tensor.

    Returns:
        (batch, 8) float tensor of spike events.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    thr = torch.tensor(thresholds, dtype=obs_batch.dtype, device=obs_batch.device)

    on_spikes = (obs_batch > thr).float()
    off_spikes = (obs_batch < -thr).float()

    # Interleave ON/OFF pairs: [on0, off0, on1, off1, ...]
    batch_size = obs_batch.shape[0]
    result = torch.zeros(batch_size, 8, dtype=obs_batch.dtype, device=obs_batch.device)
    result[:, 0::2] = on_spikes
    result[:, 1::2] = off_spikes

    if device is not None:
        result = result.to(device)
    return result
