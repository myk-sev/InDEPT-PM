"""Transformer variants with cyclical time in history and forecast inputs."""

from dataclasses import dataclass
from typing import ClassVar

from .dual_encoder_patch_transformer import (
    DualEncoderPatchTransformer,
    PatchTransformerConfig,
)
from .dual_encoder_transformer import (
    DualEncoderTransformer,
    TimestepTransformerConfig,
)


@dataclass(frozen=True)
class CyclicalPatchTransformerConfig(PatchTransformerConfig):
    cyclical_time: ClassVar[bool] = True


@dataclass(frozen=True)
class CyclicalTimestepTransformerConfig(TimestepTransformerConfig):
    cyclical_time: ClassVar[bool] = True


class CyclicalDualEncoderPatchTransformer(DualEncoderPatchTransformer):
    """Patch transformer with daily, weekly, and annual cycles."""

    history_feature_count = 10
    forecast_feature_count = 7


class CyclicalDualEncoderTransformer(DualEncoderTransformer):
    """Timestep transformer with daily, weekly, and annual cycles."""

    history_feature_count = 10
    forecast_feature_count = 7
