"""Channel-independent PatchTST adapted to the AirGuard model inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .dual_encoder_patch_transformer import (
    PatchTransformerConfig,
    _encoder,
    _missing_aware_history,
)


@dataclass(frozen=True)
class PatchTSTConfig(PatchTransformerConfig):
    """PatchTST uses the model zoo's existing patch-transformer options."""


class ChannelIndependentPatchEmbedding(nn.Module):
    """Patch every variable separately with one shared linear projection."""

    def __init__(
        self,
        sequence_length: int,
        patch_size: int,
        stride: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        remaining = max(sequence_length - patch_size, 0)
        self.patch_count = math.ceil(remaining / stride) + 1
        padded_length = (self.patch_count - 1) * stride + patch_size
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padded_length - sequence_length
        self.projection = nn.Linear(patch_size, embedding_dim)
        self.position = nn.Parameter(
            torch.empty(1, 1, self.patch_count, embedding_dim)
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.sequence_length:
            raise ValueError(
                f"expected [batch, {self.sequence_length}, channels], "
                f"received {list(values.shape)}"
            )
        values = values.transpose(1, 2)
        if self.padding:
            values = F.pad(values, (0, self.padding), mode="replicate")
        patches = values.unfold(2, self.patch_size, self.stride)
        return self.projection(patches) + self.position


class PatchTST(nn.Module):
    """Encode each input channel independently, then forecast indoor PM2.5."""

    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.config = config
        self.history_patches = ChannelIndependentPatchEmbedding(
            config.history_hours,
            config.history_patch_size,
            config.history_patch_stride,
            config.history_embedding_dim,
        )
        self.forecast_patches = ChannelIndependentPatchEmbedding(
            config.prediction_hours,
            config.forecast_patch_size,
            config.forecast_patch_stride,
            config.forecast_embedding_dim,
        )
        self.history_encoder = _encoder(config, "history")
        self.forecast_encoder = _encoder(config, "forecast")
        head_width = (
            8
            * self.history_patches.patch_count
            * config.history_embedding_dim
            + self.forecast_patches.patch_count * config.forecast_embedding_dim
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.decoder_dropout),
            nn.Linear(head_width, config.prediction_hours),
        )

    def forward(
        self, history: torch.Tensor, forecast: torch.Tensor
    ) -> torch.Tensor:
        history = _missing_aware_history(history)
        if forecast.ndim != 3 or tuple(forecast.shape[1:]) != (
            self.config.prediction_hours,
            1,
        ):
            raise ValueError(
                f"expected forecast [batch, {self.config.prediction_hours}, 1], "
                f"received {list(forecast.shape)}"
            )
        encoded = torch.cat(
            (
                self._encode(history, self.history_patches, self.history_encoder),
                self._encode(
                    forecast, self.forecast_patches, self.forecast_encoder
                ),
            ),
            dim=1,
        )
        return self.head(encoded)

    @staticmethod
    def _encode(
        values: torch.Tensor,
        embedding: ChannelIndependentPatchEmbedding,
        encoder: nn.TransformerEncoder,
    ) -> torch.Tensor:
        tokens = embedding(values)
        batch, channels, patches, width = tokens.shape
        encoded = encoder(tokens.reshape(batch * channels, patches, width))
        return encoded.reshape(batch, channels * patches * width)


Model = PatchTST

__all__ = [
    "ChannelIndependentPatchEmbedding",
    "PatchTST",
    "PatchTSTConfig",
    "Model",
]
