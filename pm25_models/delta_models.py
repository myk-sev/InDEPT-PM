"""Models that predict changes from the last indoor observation."""

import torch

from .cyclical_transformers import (
    CyclicalDualEncoderPatchTransformer,
    CyclicalDualEncoderTransformer,
)
from .patchtst import PatchTST


class _DeltaFromLastHour:
    def forward(
        self, history: torch.Tensor, forecast: torch.Tensor
    ) -> torch.Tensor:
        delta = super().forward(history, forecast)
        return delta + history[:, -1, 1].unsqueeze(1)


class DeltaCyclicalDualEncoderPatchTransformer(
    _DeltaFromLastHour, CyclicalDualEncoderPatchTransformer
):
    """Predict indoor PM2.5 changes with the cyclical patch transformer."""


class DeltaCyclicalDualEncoderTransformer(
    _DeltaFromLastHour, CyclicalDualEncoderTransformer
):
    """Predict indoor PM2.5 changes with the cyclical timestep transformer."""


class DeltaPatchTST(_DeltaFromLastHour, PatchTST):
    """Predict indoor PM2.5 changes with PatchTST."""


__all__ = [
    "DeltaCyclicalDualEncoderPatchTransformer",
    "DeltaCyclicalDualEncoderTransformer",
    "DeltaPatchTST",
]
