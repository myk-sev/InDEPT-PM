from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PatchTransformerConfig:
    history_hours: int = 168
    prediction_hours: int = 36
    history_patch_size: int = 24
    history_patch_stride: int = 12
    history_embedding_dim: int = 128
    history_heads: int = 4
    history_head_dim: int = 32
    history_layers: int = 2
    history_feedforward_dim: int = 256
    history_dropout: float = 0.1
    history_activation: str = "gelu"
    history_norm_first: bool = True
    history_layer_norm_eps: float = 1e-5
    forecast_patch_size: int = 6
    forecast_patch_stride: int = 3
    forecast_embedding_dim: int = 64
    forecast_heads: int = 4
    forecast_head_dim: int = 16
    forecast_layers: int = 2
    forecast_feedforward_dim: int = 128
    forecast_dropout: float = 0.1
    forecast_activation: str = "gelu"
    forecast_norm_first: bool = True
    forecast_layer_norm_eps: float = 1e-5
    decoder_embedding_dim: int = 128
    decoder_heads: int = 4
    decoder_head_dim: int = 32
    decoder_layers: int = 2
    decoder_feedforward_dim: int = 256
    decoder_dropout: float = 0.1
    decoder_activation: str = "gelu"
    decoder_norm_first: bool = True
    decoder_layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        _validate_transformer_config(self)
        for stream in ("history", "forecast"):
            if getattr(self, f"{stream}_patch_stride") > getattr(
                self, f"{stream}_patch_size"
            ):
                raise ValueError(
                    f"{stream}_patch_stride cannot exceed {stream}_patch_size"
                )


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        feature_count: int,
        patch_size: int,
        stride: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        remaining = max(sequence_length - patch_size, 0)
        self.patch_count = math.ceil(remaining / stride) + 1
        padded_length = (self.patch_count - 1) * stride + patch_size
        self.sequence_length = sequence_length
        self.feature_count = feature_count
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padded_length - sequence_length
        self.projection = nn.Linear(feature_count * patch_size, embedding_dim)
        self.position = nn.Parameter(
            torch.empty(1, self.patch_count, embedding_dim)
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        expected = (self.sequence_length, self.feature_count)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(
                f"expected [batch, {expected[0]}, {expected[1]}], "
                f"received {list(values.shape)}"
            )
        values = F.pad(values.transpose(1, 2), (0, self.padding))
        patches = values.unfold(2, self.patch_size, self.stride)
        patches = patches.permute(0, 2, 1, 3).flatten(2)
        return self.projection(patches) + self.position


class DualEncoderPatchTransformer(nn.Module):
    history_feature_count = 8
    forecast_feature_count = 1

    def __init__(self, config: PatchTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.history_patches = PatchEmbedding(
            config.history_hours,
            self.history_feature_count,
            config.history_patch_size,
            config.history_patch_stride,
            config.history_embedding_dim,
        )
        self.forecast_patches = PatchEmbedding(
            config.prediction_hours,
            self.forecast_feature_count,
            config.forecast_patch_size,
            config.forecast_patch_stride,
            config.forecast_embedding_dim,
        )
        self.history_encoder = _encoder(config, "history")
        self.forecast_encoder = _encoder(config, "forecast")
        self.history_projection = nn.Linear(
            config.history_embedding_dim, config.decoder_embedding_dim
        )
        self.forecast_projection = nn.Linear(
            config.forecast_embedding_dim, config.decoder_embedding_dim
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_embedding_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_feedforward_dim,
            dropout=config.decoder_dropout,
            activation=config.decoder_activation,
            layer_norm_eps=config.decoder_layer_norm_eps,
            batch_first=True,
            norm_first=config.decoder_norm_first,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            config.decoder_layers,
            norm=nn.LayerNorm(
                config.decoder_embedding_dim,
                eps=config.decoder_layer_norm_eps,
            ),
        )
        self.queries = nn.Parameter(
            torch.empty(1, config.prediction_hours, config.decoder_embedding_dim)
        )
        self.output = nn.Linear(config.decoder_embedding_dim, 1)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(
        self, history: torch.Tensor, forecast: torch.Tensor
    ) -> torch.Tensor:
        history = _missing_aware_history(history)
        history_memory = self.history_projection(
            self.history_encoder(self.history_patches(history))
        )
        forecast_memory = self.forecast_projection(
            self.forecast_encoder(self.forecast_patches(forecast))
        )
        memory = torch.cat((history_memory, forecast_memory), dim=1)
        queries = self.queries.expand(history.shape[0], -1, -1)
        return self.output(self.decoder(queries, memory)).squeeze(-1)


def _missing_aware_history(history: torch.Tensor) -> torch.Tensor:
    """Append availability and normalized recency to normalized history."""
    if history.ndim != 3 or history.shape[2] not in {6, 8}:
        raise ValueError(
            "history must contain outdoor PM2.5, indoor PM2.5, and time features"
        )

    available = torch.isfinite(history[..., :1])
    steps = torch.arange(
        history.shape[1], device=history.device, dtype=torch.int64
    ).view(1, -1, 1)
    last_seen = torch.where(available, steps, -history.shape[1])
    last_seen = torch.cummax(last_seen, dim=1).values
    recency = (steps - last_seen).clamp(max=history.shape[1])
    recency = recency.to(history.dtype) / history.shape[1]
    outdoor = torch.where(available, history[..., :1], 0.0)
    return torch.cat(
        (outdoor, history[..., 1:], available.to(history.dtype), recency),
        dim=2,
    )


def _encoder(
    config: PatchTransformerConfig, stream: str
) -> nn.TransformerEncoder:
    embedding = getattr(config, f"{stream}_embedding_dim")
    epsilon = getattr(config, f"{stream}_layer_norm_eps")
    layer = nn.TransformerEncoderLayer(
        d_model=embedding,
        nhead=getattr(config, f"{stream}_heads"),
        dim_feedforward=getattr(config, f"{stream}_feedforward_dim"),
        dropout=getattr(config, f"{stream}_dropout"),
        activation=getattr(config, f"{stream}_activation"),
        layer_norm_eps=epsilon,
        batch_first=True,
        norm_first=getattr(config, f"{stream}_norm_first"),
    )
    return nn.TransformerEncoder(
        layer,
        getattr(config, f"{stream}_layers"),
        norm=nn.LayerNorm(embedding, eps=epsilon),
        enable_nested_tensor=False,
    )


def _validate_transformer_config(config: PatchTransformerConfig) -> None:
    positive = (
        "history_hours",
        "prediction_hours",
        "history_patch_size",
        "history_patch_stride",
        "history_layers",
        "history_feedforward_dim",
        "forecast_patch_size",
        "forecast_patch_stride",
        "forecast_layers",
        "forecast_feedforward_dim",
        "decoder_layers",
        "decoder_feedforward_dim",
    )
    for name in positive:
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    for stream in ("history", "forecast", "decoder"):
        embedding = getattr(config, f"{stream}_embedding_dim")
        heads = getattr(config, f"{stream}_heads")
        head_dim = getattr(config, f"{stream}_head_dim")
        if embedding < 1 or heads < 1 or head_dim < 1:
            raise ValueError(
                f"{stream} embedding, heads, and head dimension must be positive"
            )
        if embedding != heads * head_dim:
            raise ValueError(
                f"{stream}_embedding_dim ({embedding}) must equal "
                f"{stream}_heads ({heads}) x {stream}_head_dim ({head_dim})"
            )
        dropout = getattr(config, f"{stream}_dropout")
        if not 0 <= dropout < 1:
            raise ValueError(f"{stream}_dropout must be in [0, 1)")
        if getattr(config, f"{stream}_activation") not in {"relu", "gelu"}:
            raise ValueError(f"{stream}_activation must be relu or gelu")
        if getattr(config, f"{stream}_layer_norm_eps") <= 0:
            raise ValueError(f"{stream}_layer_norm_eps must be positive")
