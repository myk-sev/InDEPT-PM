from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    history_hours: int = 168
    input_features: int = 8
    output_channels: int = 2
    model_dim: int = 64
    layers: int = 3
    heads: int = 4
    dropout: float = 0.1


Builder = Callable[[ModelConfig], nn.Module]
_MODELS: dict[str, Builder] = {}


def register_model(name: str) -> Callable[[Builder], Builder]:
    """Register a reconstructor that accepts ModelConfig and returns [B, T, 2]."""
    if not name:
        raise ValueError("model name cannot be empty")

    def decorate(builder: Builder) -> Builder:
        if name in _MODELS:
            raise ValueError(f"duplicate model name: {name}")
        _MODELS[name] = builder
        return builder

    return decorate


def build_model(name: str, config: ModelConfig) -> nn.Module:
    try:
        model = _MODELS[name](config)
    except KeyError as error:
        raise ValueError(f"unknown model {name!r}; choose from {model_names()}") from error
    return model


def model_names() -> tuple[str, ...]:
    return tuple(_MODELS)


class TransformerReconstructor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.model_dim % config.heads:
            raise ValueError("model_dim must be divisible by heads")
        self.config = config
        self.input_projection = nn.Linear(config.input_features, config.model_dim)
        self.position = nn.Parameter(
            torch.empty(1, config.history_hours, config.model_dim)
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            config.model_dim,
            config.heads,
            config.model_dim * 4,
            config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.reconstruction_head = nn.Linear(config.model_dim, config.output_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_features(features, self.config)
        encoded = self.encoder(self.input_projection(features) + self.position)
        return self.reconstruction_head(encoded)


class GruReconstructor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_features, config.model_dim)
        self.encoder = nn.GRU(
            config.model_dim,
            config.model_dim,
            config.layers,
            dropout=config.dropout if config.layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.reconstruction_head = nn.Linear(
            config.model_dim * 2, config.output_channels
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_features(features, self.config)
        encoded, _ = self.encoder(self.input_projection(features))
        return self.reconstruction_head(encoded)


def _validate_features(features: torch.Tensor, config: ModelConfig) -> None:
    expected = (config.history_hours, config.input_features)
    if features.ndim != 3 or tuple(features.shape[1:]) != expected:
        raise ValueError(
            f"expected features [batch, {expected[0]}, {expected[1]}], "
            f"received {list(features.shape)}"
        )


register_model("transformer")(TransformerReconstructor)
register_model("gru")(GruReconstructor)

__all__ = [
    "GruReconstructor",
    "ModelConfig",
    "TransformerReconstructor",
    "build_model",
    "model_names",
    "register_model",
]
