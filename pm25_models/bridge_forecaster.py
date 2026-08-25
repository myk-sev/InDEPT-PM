"""Forecast models that reuse a completed masked-reconstruction history encoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import torch
from torch import nn

from masked_pretraining.masking import (
    MASK_SENTINEL,
    STAGES,
    TEMPO_BRIDGE_MISSING_FRACTIONS,
    TEMPO_BRIDGE_STAGES,
)
from masked_pretraining.models import (
    ModelConfig as HistoryConfig,
    build_model as build_history_model,
    canonical_model_name,
    model_names as history_model_names,
)


BRIDGE_FORECAST_PREFIX = "bridge-forecast-"
FORECAST_FEATURES = 7


def bridge_forecast_name(history_model_name: str) -> str:
    return BRIDGE_FORECAST_PREFIX + canonical_model_name(history_model_name)


def bridge_forecast_names() -> tuple[str, ...]:
    return tuple(bridge_forecast_name(name) for name in history_model_names())


def bridge_history_model_name(forecast_model_name: str) -> str:
    if not forecast_model_name.startswith(BRIDGE_FORECAST_PREFIX):
        raise ValueError(f"not a bridge forecast model: {forecast_model_name}")
    name = forecast_model_name.removeprefix(BRIDGE_FORECAST_PREFIX)
    if name not in history_model_names():
        raise ValueError(f"unknown bridge history model: {name}")
    return name


@dataclass(frozen=True)
class BridgeForecastConfig:
    history_hours: int = 168
    prediction_hours: int = 36
    model_dim: int = 64
    layers: int = 3
    heads: int = 4
    dropout: float = 0.1
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
    cyclical_time: ClassVar[bool] = True

    def __post_init__(self) -> None:
        positive = (
            "history_hours",
            "prediction_hours",
            "model_dim",
            "layers",
            "heads",
            "forecast_embedding_dim",
            "forecast_heads",
            "forecast_head_dim",
            "forecast_layers",
            "forecast_feedforward_dim",
            "decoder_embedding_dim",
            "decoder_heads",
            "decoder_head_dim",
            "decoder_layers",
            "decoder_feedforward_dim",
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        for stream in ("forecast", "decoder"):
            embedding = getattr(self, f"{stream}_embedding_dim")
            if embedding != getattr(self, f"{stream}_heads") * getattr(
                self, f"{stream}_head_dim"
            ):
                raise ValueError(
                    f"{stream}_embedding_dim must equal {stream}_heads x "
                    f"{stream}_head_dim"
                )
            if not 0 <= getattr(self, f"{stream}_dropout") < 1:
                raise ValueError(f"{stream}_dropout must be in [0, 1)")
            if getattr(self, f"{stream}_activation") not in {"relu", "gelu"}:
                raise ValueError(f"{stream}_activation must be relu or gelu")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def history_config(self) -> HistoryConfig:
        return HistoryConfig(
            history_hours=self.history_hours,
            model_dim=self.model_dim,
            layers=self.layers,
            heads=self.heads,
            dropout=self.dropout,
        )


class BridgeForecaster(nn.Module):
    """Use a reconstruction encoder as memory for a future-NAQFC decoder."""

    def __init__(self, history_model_name: str, config: BridgeForecastConfig) -> None:
        super().__init__()
        self.config = config
        self.history_model_name = canonical_model_name(history_model_name)
        self.history = build_history_model(self.history_model_name, config.history_config)
        self.history.reconstruction_head = nn.Identity()
        history_width = config.model_dim * (2 if self.history_model_name == "gru" else 1)
        self.history_projection = nn.Linear(history_width, config.decoder_embedding_dim)
        self.forecast_projection = nn.Linear(
            FORECAST_FEATURES, config.forecast_embedding_dim
        )
        self.forecast_position = nn.Parameter(
            torch.empty(1, config.prediction_hours, config.forecast_embedding_dim)
        )
        self.forecast_encoder = _encoder(config, "forecast")
        self.forecast_to_decoder = nn.Linear(
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
                config.decoder_embedding_dim, eps=config.decoder_layer_norm_eps
            ),
        )
        self.output = nn.Linear(config.decoder_embedding_dim, 1)
        nn.init.trunc_normal_(self.forecast_position, std=0.02)

    def forward(self, history: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        history = _bridge_history(history, self.config.history_hours)
        expected = (self.config.prediction_hours, FORECAST_FEATURES)
        if forecast.ndim != 3 or tuple(forecast.shape[1:]) != expected:
            raise ValueError(
                f"expected forecast [batch, {expected[0]}, {expected[1]}], "
                f"received {list(forecast.shape)}"
            )
        memory = self.history_projection(self.history.encode(history))
        future = self.forecast_encoder(
            self.forecast_projection(forecast) + self.forecast_position
        )
        decoded = self.decoder(self.forecast_to_decoder(future), memory)
        return self.output(decoded).squeeze(-1)

    def history_parameters(self):
        return self.history.parameters()

    def load_pretrained_history(self, checkpoint: dict) -> tuple[str, ...]:
        validate_bridge_checkpoint(checkpoint, self.history_model_name, self.config)
        source = {
            name: value
            for name, value in checkpoint["model_state"].items()
            if not name.startswith("reconstruction_head.")
        }
        expected = self.history.state_dict()
        if source.keys() != expected.keys():
            missing = sorted(expected.keys() - source.keys())
            unexpected = sorted(source.keys() - expected.keys())
            raise ValueError(
                "bridge checkpoint transfer keys do not match: "
                f"missing={missing}, unexpected={unexpected}"
            )
        shape_mismatches = [
            name
            for name, value in source.items()
            if value.shape != expected[name].shape
        ]
        if shape_mismatches:
            raise ValueError(
                "bridge checkpoint transfer shapes do not match: "
                + ", ".join(shape_mismatches)
            )
        self.history.load_state_dict(source)
        return tuple(source)


def validate_bridge_checkpoint(
    checkpoint: dict,
    history_model_name: str | None = None,
    config: BridgeForecastConfig | None = None,
) -> dict:
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state"), dict
    ) or not isinstance(checkpoint.get("metadata"), dict):
        raise ValueError("invalid bridge checkpoint")
    metadata = checkpoint["metadata"]
    source_name = canonical_model_name(str(metadata.get("model_name", "")))
    if source_name not in history_model_names():
        raise ValueError("bridge checkpoint has an unknown history model")
    if history_model_name and source_name != canonical_model_name(history_model_name):
        raise ValueError(
            f"bridge checkpoint model is {source_name}, expected {history_model_name}"
        )
    completed = set(metadata.get("completed_stages", ()))
    required = set(STAGES + TEMPO_BRIDGE_STAGES)
    if metadata.get("stage") != TEMPO_BRIDGE_STAGES[-1] or not required <= completed:
        raise ValueError("bridge checkpoint did not complete the full curriculum")
    bridge = metadata.get("masking", {}).get("tempo_missingness_bridge", {})
    if not bridge.get("enabled") or not bridge.get("synthetic_only"):
        raise ValueError("bridge checkpoint is missing synthetic bridge metadata")
    if bridge.get("tempo_data_used") is not False:
        raise ValueError("bridge checkpoint must not claim TEMPO training data")
    if bridge.get("outdoor_artificial_missing_fractions") != dict(
        TEMPO_BRIDGE_MISSING_FRACTIONS
    ):
        raise ValueError("bridge checkpoint has invalid missingness fractions")
    transfer = metadata.get("transfer", {})
    if transfer.get("input_feature_order") != [
        "outdoor_value",
        "indoor_value",
        "daily_sin",
        "daily_cos",
        "weekly_sin",
        "weekly_cos",
        "annual_sin",
        "annual_cos",
    ]:
        raise ValueError("bridge checkpoint has an incompatible feature order")
    if metadata.get("masking", {}).get("sentinel") != MASK_SENTINEL:
        raise ValueError("bridge checkpoint has an incompatible missing sentinel")
    if config and metadata.get("model_config") != asdict(config.history_config):
        raise ValueError("bridge checkpoint history configuration does not match")
    normalizer = metadata.get("normalizer", {})
    if not _pair(normalizer.get("mean")) or not _pair(
        normalizer.get("standard_deviation"), positive=True
    ):
        raise ValueError("bridge checkpoint has invalid normalization metadata")
    return metadata


def load_bridge_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"bridge checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_bridge_checkpoint(checkpoint)
    return checkpoint


def bridge_config_values(checkpoint: dict) -> dict[str, int | float]:
    metadata = validate_bridge_checkpoint(checkpoint)
    source = metadata["model_config"]
    return {
        "history_hours": source["history_hours"],
        "model_dim": source["model_dim"],
        "layers": source["layers"],
        "heads": source["heads"],
        "dropout": source["dropout"],
    }


def _bridge_history(history: torch.Tensor, history_hours: int) -> torch.Tensor:
    expected = (history_hours, 8)
    if history.ndim != 3 or tuple(history.shape[1:]) != expected:
        raise ValueError(
            f"expected history [batch, {expected[0]}, {expected[1]}], "
            f"received {list(history.shape)}"
        )
    if not torch.isfinite(history[..., 1:]).all():
        raise ValueError("indoor PM2.5 and time history features must be complete")
    outdoor = history[..., :1]
    return torch.cat(
        (torch.where(torch.isfinite(outdoor), outdoor, MASK_SENTINEL), history[..., 1:]),
        dim=2,
    )


def _encoder(config: BridgeForecastConfig, stream: str) -> nn.TransformerEncoder:
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


def _pair(value: object, positive: bool = False) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
        and (not positive or all(item > 0 for item in value))
    )


__all__ = [
    "BRIDGE_FORECAST_PREFIX",
    "BridgeForecastConfig",
    "BridgeForecaster",
    "bridge_config_values",
    "bridge_forecast_name",
    "bridge_forecast_names",
    "bridge_history_model_name",
    "load_bridge_checkpoint",
    "validate_bridge_checkpoint",
]
