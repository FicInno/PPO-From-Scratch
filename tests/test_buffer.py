import numpy as np
import torch

from utils.buffer import RolloutBuffer


def test_buffer_shapes_and_advantage_normalization():
    buffer = RolloutBuffer(capacity=4, observation_dim=3, action_dim=1, discrete=False)
    for index in range(4):
        buffer.add(np.ones(3) * index, np.array([0.1]), 1.0, index == 3, index == 3, 0.0, 0.0, -0.2)
    batch = buffer.finish(0.99, 0.95, True, torch.device("cpu"))
    assert batch.observations.shape == (4, 3)
    assert batch.actions.shape == (4, 1)
    assert abs(float(batch.advantages.mean())) < 1e-6
    assert torch.isfinite(batch.returns).all()

