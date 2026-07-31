"""Dual-encoder PM2.5 transformer with one token per hourly timestep."""

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from pm25_models import ModelConfig


class TimestepEmbedding(nn.Module):
    def __init__(
        self, sequence_length: int, feature_count: int, embedding_dim: int
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_count = feature_count
        self.projection = nn.Linear(feature_count, embedding_dim)
        self.position = nn.Parameter(
            torch.empty(1, sequence_length, embedding_dim)
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        expected = (self.sequence_length, self.feature_count)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(
                f"expected [batch, {expected[0]}, {expected[1]}], "
                f"received {list(values.shape)}"
            )
        return self.projection(values) + self.position


class DualEncoderTransformer(nn.Module):
    """Predict future indoor PM2.5 without grouping inputs into patches."""

    def __init__(self, config: "ModelConfig") -> None:
        super().__init__()
        self.config = config
        self.history_embedding = TimestepEmbedding(
            config.history_hours, 8, config.history_embedding_dim
        )
        self.forecast_embedding = TimestepEmbedding(
            config.prediction_hours, 1, config.forecast_embedding_dim
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
            self.history_encoder(self.history_embedding(history))
        )
        forecast_memory = self.forecast_projection(
            self.forecast_encoder(self.forecast_embedding(forecast))
        )
        memory = torch.cat((history_memory, forecast_memory), dim=1)
        queries = self.queries.expand(history.shape[0], -1, -1)
        return self.output(self.decoder(queries, memory)).squeeze(-1)


def _missing_aware_history(history: torch.Tensor) -> torch.Tensor:
    if history.ndim != 3 or history.shape[2] != 6:
        raise ValueError(
            "history must contain outdoor PM2.5, indoor PM2.5, and four time features"
        )

    available = torch.isfinite(history[..., :1])
    steps = torch.arange(
        history.shape[1], device=history.device, dtype=torch.int64
    ).view(1, -1, 1)
    last_seen = torch.where(available, steps, -history.shape[1])
    recency = steps - torch.cummax(last_seen, dim=1).values
    outdoor = torch.where(available, history[..., :1], 0.0)
    return torch.cat(
        (
            outdoor,
            history[..., 1:],
            available.to(history.dtype),
            recency.clamp(max=history.shape[1]).to(history.dtype)
            / history.shape[1],
        ),
        dim=2,
    )


def _encoder(config: "ModelConfig", stream: str) -> nn.TransformerEncoder:
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


Model = DualEncoderTransformer

__all__ = ["DualEncoderTransformer", "Model"]
