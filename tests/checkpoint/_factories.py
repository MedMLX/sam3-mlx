"""Shared tiny module factories for checkpoint conversion tests."""

import mlx.core as mx
from mlx import nn


class WeightLeaf(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.weight = mx.zeros(shape)
