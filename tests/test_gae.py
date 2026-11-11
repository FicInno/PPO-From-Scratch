import numpy as np

from utils.gae import compute_gae


def test_gae_matches_hand_calculation():
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.25], dtype=np.float32)
    next_values = np.array([0.25, 0.0], dtype=np.float32)
    terminated = np.array([0.0, 1.0], dtype=np.float32)
    episode_ends = np.array([0.0, 1.0], dtype=np.float32)
    advantages, returns = compute_gae(rewards, values, next_values, terminated, episode_ends, 0.9, 0.8)
    expected_last = 1.0 - 0.25
    expected_first = 1.0 + 0.9 * 0.25 - 0.5 + 0.9 * 0.8 * expected_last
    np.testing.assert_allclose(advantages, [expected_first, expected_last], rtol=1e-6)
    np.testing.assert_allclose(returns, advantages + values, rtol=1e-6)


def test_truncation_bootstraps_but_breaks_recursion():
    advantages, _ = compute_gae(
        np.array([1.0, 100.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([2.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.95,
    )
    assert np.isclose(advantages[0], 2.8)

