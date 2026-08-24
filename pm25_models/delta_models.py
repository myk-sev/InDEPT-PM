"""Models that predict indoor PM2.5 changes."""

import torch

from .cyclical_transformers import (
    CyclicalDualEncoderPatchTransformer,
    CyclicalDualEncoderTransformer,
)
from .patchtst import CyclicalPatchTST, PatchTST


class _StepDeltaFromLastHour:
    def forward(
        self, history: torch.Tensor, forecast: torch.Tensor
    ) -> torch.Tensor:
        delta = super().forward(history, forecast)
        return delta.cumsum(dim=1) + history[:, -1, 1].unsqueeze(1)


class _AbsoluteDeltaFromLastHour:
    def forward(
        self, history: torch.Tensor, forecast: torch.Tensor
    ) -> torch.Tensor:
        delta = super().forward(history, forecast)
        return delta + history[:, -1, 1].unsqueeze(1)


class DeltaCyclicalDualEncoderPatchTransformer(
    _StepDeltaFromLastHour, CyclicalDualEncoderPatchTransformer
):
    """Predict cumulative hourly PM2.5 changes with the cyclical transformer."""


class DeltaCyclicalDualEncoderTransformer(
    _StepDeltaFromLastHour, CyclicalDualEncoderTransformer
):
    """Predict cumulative hourly PM2.5 changes with the timestep transformer."""


class DeltaPatchTST(_StepDeltaFromLastHour, PatchTST):
    """Predict cumulative hourly PM2.5 changes with PatchTST."""


class DeltaCyclicalPatchTST(_StepDeltaFromLastHour, CyclicalPatchTST):
    """Predict cumulative hourly PM2.5 changes with cyclical PatchTST."""


class AbsoluteDeltaCyclicalDualEncoderPatchTransformer(
    _AbsoluteDeltaFromLastHour, CyclicalDualEncoderPatchTransformer
):
    """Preserve the original horizon-wise delta behavior."""


class AbsoluteDeltaCyclicalDualEncoderTransformer(
    _AbsoluteDeltaFromLastHour, CyclicalDualEncoderTransformer
):
    """Preserve the original horizon-wise delta behavior."""


class AbsoluteDeltaPatchTST(_AbsoluteDeltaFromLastHour, PatchTST):
    """Preserve the original horizon-wise delta behavior."""


class AbsoluteDeltaCyclicalPatchTST(_AbsoluteDeltaFromLastHour, CyclicalPatchTST):
    """Preserve the original horizon-wise delta behavior."""


__all__ = [
    "AbsoluteDeltaCyclicalDualEncoderPatchTransformer",
    "AbsoluteDeltaCyclicalDualEncoderTransformer",
    "AbsoluteDeltaCyclicalPatchTST",
    "AbsoluteDeltaPatchTST",
    "DeltaCyclicalDualEncoderPatchTransformer",
    "DeltaCyclicalDualEncoderTransformer",
    "DeltaCyclicalPatchTST",
    "DeltaPatchTST",
]
