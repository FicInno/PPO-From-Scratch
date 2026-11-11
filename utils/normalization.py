"""Running observation statistics used by training and evaluation."""

from pathlib import Path

import numpy as np


class ObservationNormalizer:
    """Numerically stable running mean and variance for vector observations."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4, clip: float = 10.0) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.variance = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)
        self.clip = float(clip)

    def update(self, observations: np.ndarray) -> None:
        batch = np.asarray(observations, dtype=np.float64).reshape(-1, *self.mean.shape)
        batch_mean = batch.mean(axis=0)
        batch_variance = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        old_sum = self.variance * self.count
        batch_sum = batch_variance * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.variance = (old_sum + batch_sum + correction) / total_count
        self.count = float(total_count)

    def normalize(self, observation: np.ndarray) -> np.ndarray:
        value = (np.asarray(observation, dtype=np.float32) - self.mean) / np.sqrt(self.variance + 1e-8)
        return np.clip(value, -self.clip, self.clip).astype(np.float32)

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            mean=self.mean,
            variance=self.variance,
            count=np.asarray(self.count),
            clip=np.asarray(self.clip),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ObservationNormalizer":
        data = np.load(path)
        normalizer = cls(tuple(data["mean"].shape), clip=float(data["clip"]))
        normalizer.mean = data["mean"].astype(np.float64)
        normalizer.variance = data["variance"].astype(np.float64)
        normalizer.count = float(data["count"])
        return normalizer

