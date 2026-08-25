from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from .masking import MASK_SENTINEL


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
DEFAULT_MODEL = "single-self-attention-encoder"
_MODEL_ALIASES = {"transformer": DEFAULT_MODEL}


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
    name = canonical_model_name(name)
    try:
        model = _MODELS[name](config)
    except KeyError as error:
        raise ValueError(f"unknown model {name!r}; choose from {model_names()}") from error
    return model


def model_names() -> tuple[str, ...]:
    return tuple(_MODELS)


def canonical_model_name(name: str) -> str:
    return _MODEL_ALIASES.get(name, name)


class SingleSelfAttentionEncoder(nn.Module):
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


class _PairedReconstructor(nn.Module):
    def _reconstruct(
        self, outdoor: torch.Tensor, indoor: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (
                self.reconstruction_head["outdoor"](outdoor),
                self.reconstruction_head["indoor"](indoor),
            ),
            dim=-1,
        )


class _DualEncoderBackbone(_PairedReconstructor):
    outdoor_availability = False
    outdoor_recency = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        _validate_stream_config(config)
        self.config = config
        stream_features = config.input_features - 1
        self.position = nn.Parameter(
            torch.empty(1, config.history_hours, config.model_dim)
        )
        self.input_projection = nn.ModuleDict(
            {
                "indoor": nn.Linear(stream_features, config.model_dim),
                "outdoor": nn.Linear(
                    stream_features
                    + int(self.outdoor_availability)
                    + int(self.outdoor_recency),
                    config.model_dim,
                ),
            }
        )
        self.encoder = nn.ModuleDict(
            {
                "indoor": _encoder(config),
                "outdoor": _encoder(config),
            }
        )
        self.reconstruction_head = _reconstruction_heads(config)
        nn.init.trunc_normal_(self.position, std=0.02)

    def _encode_streams(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_features(features, self.config)
        time = features[..., 2:]
        indoor = torch.cat((features[..., 1:2], time), dim=-1)
        outdoor = torch.cat(
            (
                _outdoor_stream(
                    features, self.outdoor_availability, self.outdoor_recency
                ),
                time,
            ),
            dim=-1,
        )
        indoor = self.encoder["indoor"](
            self.input_projection["indoor"](indoor) + self.position
        )
        outdoor = self.encoder["outdoor"](
            self.input_projection["outdoor"](outdoor) + self.position
        )
        return outdoor, indoor


class DualEncoderSelfFusion(_DualEncoderBackbone):
    """Encode indoor/time and outdoor/time separately, then fuse jointly."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.encoder["fusion"] = _encoder(config, layers=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outdoor, indoor = self._encode_streams(features)
        outdoor, indoor = self.encoder["fusion"](
            torch.cat((outdoor, indoor), dim=1)
        ).split(self.config.history_hours, dim=1)
        return self._reconstruct(outdoor, indoor)


class DualEncoderCrossFusion(_DualEncoderBackbone):
    """Encode indoor/time and outdoor/time separately, then fuse across streams."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.encoder["fusion"] = _CrossAttentionFusion(config)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outdoor, indoor = self._encode_streams(features)
        return self._reconstruct(
            self.encoder["fusion"](outdoor, indoor),
            self.encoder["fusion"](indoor, outdoor),
        )


class DualEncoderSelfFusionOutdoorAvailability(DualEncoderSelfFusion):
    """Dual self-fusion with outdoor PM2.5 and availability encoded together."""

    outdoor_availability = True


class DualEncoderCrossFusionOutdoorAvailability(DualEncoderCrossFusion):
    """Dual cross-fusion with outdoor PM2.5 and availability encoded together."""

    outdoor_availability = True


class DualEncoderSelfFusionOutdoorAvailabilityRecency(
    DualEncoderSelfFusionOutdoorAvailability
):
    """Dual self-fusion with outdoor value, availability, and recency."""

    outdoor_recency = True


class DualEncoderCrossFusionOutdoorAvailabilityRecency(
    DualEncoderCrossFusionOutdoorAvailability
):
    """Dual cross-fusion with outdoor value, availability, and recency."""

    outdoor_recency = True


class SeparateStreamSelfFusion(_PairedReconstructor):
    """Encode time, indoor PM2.5, and outdoor PM2.5 as separate streams."""

    outdoor_availability = False
    outdoor_recency = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        _validate_stream_config(config)
        self.config = config
        self.position = nn.Parameter(
            torch.empty(1, config.history_hours, config.model_dim)
        )
        self.input_projection = nn.ModuleDict(
            {
                "time": nn.Linear(config.input_features - 2, config.model_dim),
                "indoor": nn.Linear(1, config.model_dim),
                "outdoor": nn.Linear(
                    1
                    + int(self.outdoor_availability)
                    + int(self.outdoor_recency),
                    config.model_dim,
                ),
            }
        )
        self.encoder = nn.ModuleDict(
            {
                "time": _encoder(config),
                "indoor": _encoder(config),
                "outdoor": _encoder(config),
                "fusion": _encoder(config, layers=1),
            }
        )
        self.reconstruction_head = _reconstruction_heads(config)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_features(features, self.config)
        streams = (
            self.encoder[name](self.input_projection[name](values) + self.position)
            for name, values in (
                ("time", features[..., 2:]),
                ("indoor", features[..., 1:2]),
                (
                    "outdoor",
                    _outdoor_stream(
                        features,
                        self.outdoor_availability,
                        self.outdoor_recency,
                    ),
                ),
            )
        )
        _, indoor, outdoor = self.encoder["fusion"](
            torch.cat(tuple(streams), dim=1)
        ).split(self.config.history_hours, dim=1)
        return self._reconstruct(outdoor, indoor)


class SeparateStreamSelfFusionOutdoorAvailability(SeparateStreamSelfFusion):
    """Separate self-fusion with outdoor PM2.5 and availability encoded together."""

    outdoor_availability = True


class SeparateStreamSelfFusionOutdoorAvailabilityRecency(
    SeparateStreamSelfFusionOutdoorAvailability
):
    """Separate self-fusion with outdoor value, availability, and recency."""

    outdoor_recency = True


class _CrossAttentionFusion(nn.Module):
    """Fuse one encoded stream with another without additional self-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.model_dim, config.heads, config.dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim * 4, config.model_dim),
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, query: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.attention(query, context, context, need_weights=False)
        query = self.attention_norm(query + self.dropout(attended))
        return self.output_norm(query + self.dropout(self.feedforward(query)))


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


def _validate_stream_config(config: ModelConfig) -> None:
    if config.model_dim % config.heads:
        raise ValueError("model_dim must be divisible by heads")
    if config.input_features < 3:
        raise ValueError(
            "stream models require two PM2.5 and at least one time feature"
        )
    if config.output_channels != 2:
        raise ValueError("stream models require indoor and outdoor output channels")


def _outdoor_stream(
    features: torch.Tensor,
    include_availability: bool,
    include_recency: bool,
) -> torch.Tensor:
    outdoor = features[..., :1]
    if not include_availability and not include_recency:
        return outdoor
    available = outdoor.ne(MASK_SENTINEL)
    streams = [outdoor]
    if include_availability:
        streams.append(available.to(outdoor.dtype))
    if include_recency:
        steps = torch.arange(
            outdoor.shape[1], device=outdoor.device, dtype=torch.int64
        ).view(1, -1, 1)
        last_seen = torch.where(available, steps, -outdoor.shape[1])
        recency = steps - torch.cummax(last_seen, dim=1).values
        streams.append(
            recency.clamp(max=outdoor.shape[1]).to(outdoor.dtype)
            / outdoor.shape[1]
        )
    return torch.cat(tuple(streams), dim=-1)


def _encoder(config: ModelConfig, layers: int | None = None) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        config.model_dim,
        config.heads,
        config.model_dim * 4,
        config.dropout,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, config.layers if layers is None else layers)


def _reconstruction_heads(config: ModelConfig) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "indoor": nn.Linear(config.model_dim, 1),
            "outdoor": nn.Linear(config.model_dim, 1),
        }
    )


register_model(DEFAULT_MODEL)(SingleSelfAttentionEncoder)
register_model("dual-encoder-self-fusion")(DualEncoderSelfFusion)
register_model("dual-encoder-self-fusion-outdoor-availability")(
    DualEncoderSelfFusionOutdoorAvailability
)
register_model("dual-encoder-self-fusion-outdoor-availability-recency")(
    DualEncoderSelfFusionOutdoorAvailabilityRecency
)
register_model("dual-encoder-cross-fusion")(DualEncoderCrossFusion)
register_model("dual-encoder-cross-fusion-outdoor-availability")(
    DualEncoderCrossFusionOutdoorAvailability
)
register_model("dual-encoder-cross-fusion-outdoor-availability-recency")(
    DualEncoderCrossFusionOutdoorAvailabilityRecency
)
register_model("separate-stream-self-fusion")(SeparateStreamSelfFusion)
register_model("separate-stream-self-fusion-outdoor-availability")(
    SeparateStreamSelfFusionOutdoorAvailability
)
register_model("separate-stream-self-fusion-outdoor-availability-recency")(
    SeparateStreamSelfFusionOutdoorAvailabilityRecency
)
register_model("gru")(GruReconstructor)

__all__ = [
    "DEFAULT_MODEL",
    "DualEncoderCrossFusion",
    "DualEncoderCrossFusionOutdoorAvailability",
    "DualEncoderCrossFusionOutdoorAvailabilityRecency",
    "DualEncoderSelfFusion",
    "DualEncoderSelfFusionOutdoorAvailability",
    "DualEncoderSelfFusionOutdoorAvailabilityRecency",
    "GruReconstructor",
    "ModelConfig",
    "SeparateStreamSelfFusion",
    "SeparateStreamSelfFusionOutdoorAvailability",
    "SeparateStreamSelfFusionOutdoorAvailabilityRecency",
    "SingleSelfAttentionEncoder",
    "build_model",
    "canonical_model_name",
    "model_names",
    "register_model",
]
