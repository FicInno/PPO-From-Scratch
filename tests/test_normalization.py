import numpy as np

from utils.normalization import ObservationNormalizer


def test_observation_normalizer_roundtrip(tmp_path):
    normalizer = ObservationNormalizer((2,))
    normalizer.update(np.array([[1.0, 10.0], [3.0, 14.0]], dtype=np.float32))
    normalized = normalizer.normalize(np.array([2.0, 12.0], dtype=np.float32))
    np.testing.assert_allclose(normalized, np.zeros(2), atol=5e-4)

    path = tmp_path / "normalizer.npz"
    normalizer.save(path)
    restored = ObservationNormalizer.load(path)
    np.testing.assert_allclose(restored.normalize([1.0, 10.0]), normalizer.normalize([1.0, 10.0]))
