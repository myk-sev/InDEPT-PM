"""Transformer with separate stream encoders and two fusion stages."""

import torch
from torch import nn

from .dual_encoder_transformer import (
    TimestepEmbedding,
    TimestepTransformerConfig,
    _encoder,
    _missing_aware_history,
)


class _SeparateStreamBackbone(nn.Module):
    def __init__(self, config: TimestepTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.indoor_embedding = TimestepEmbedding(
            config.history_hours, 5, config.history_embedding_dim
        )
        self.outdoor_embedding = TimestepEmbedding(
            config.history_hours, 7, config.history_embedding_dim
        )
        self.forecast_embedding = TimestepEmbedding(
            config.prediction_hours, 1, config.forecast_embedding_dim
        )
        self.indoor_encoder = _encoder(config, "history")
        self.outdoor_encoder = _encoder(config, "history")
        self.forecast_encoder = _encoder(config, "forecast")

        self.history_stream = nn.Parameter(
            torch.empty(2, 1, config.history_embedding_dim)
        )
        self.history_fusion = _encoder(config, "history")
        self.history_projection = nn.Linear(
            config.history_embedding_dim, config.decoder_embedding_dim
        )
        self.forecast_projection = nn.Linear(
            config.forecast_embedding_dim, config.decoder_embedding_dim
        )
        self.output_norm = nn.LayerNorm(
            config.decoder_embedding_dim, eps=config.decoder_layer_norm_eps
        )
        self.output = nn.Linear(config.decoder_embedding_dim, 1)
        nn.init.trunc_normal_(self.history_stream, std=0.02)

    def _encode_streams(
        self, history: torch.Tensor, forecast: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 3 or history.shape[2] != 6:
            raise ValueError(
                "history must contain outdoor PM2.5, indoor PM2.5, "
                "and four time features"
            )
        history = _missing_aware_history(history)
        time = history[..., 2:-2]
        indoor = torch.cat((history[..., 1:2], time), dim=2)
        outdoor = torch.cat((history[..., :1], time, history[..., -2:]), dim=2)

        indoor = self.indoor_encoder(self.indoor_embedding(indoor))
        outdoor = self.outdoor_encoder(self.outdoor_embedding(outdoor))
        historical = torch.cat(
            (
                indoor + self.history_stream[0],
                outdoor + self.history_stream[1],
            ),
            dim=1,
        )
        historical = self.history_projection(self.history_fusion(historical))
        forecast = self.forecast_projection(
            self.forecast_encoder(self.forecast_embedding(forecast))
        )
        return historical, forecast

    def _predict(self, forecast: torch.Tensor) -> torch.Tensor:
        return self.output(self.output_norm(forecast)).squeeze(-1)


class SeparateStreamCrossFusionTransformer(_SeparateStreamBackbone):
    """Fuse histories with self-attention, then NAQFC with cross-attention."""

    def __init__(self, config: TimestepTransformerConfig) -> None:
        super().__init__(config)
        self.cross_fusion = nn.ModuleList(
            _CrossAttentionLayer(config) for _ in range(config.decoder_layers)
        )

    def forward(
        self, history: torch.Tensor, forecast: torch.Tensor
    ) -> torch.Tensor:
        historical, forecast = self._encode_streams(history, forecast)
        for layer in self.cross_fusion:
            forecast = layer(forecast, historical)
        return self._predict(forecast)


class SeparateStreamSelfFusionTransformer(_SeparateStreamBackbone):
    """Fuse histories first, then fuse history and NAQFC with self-attention."""

    def __init__(self, config: TimestepTransformerConfig) -> None:
        super().__init__(config)
        self.self_fusion = _encoder(config, "decoder")

    def forward(
        self, history: torch.Tensor, forecast: torch.Tensor
    ) -> torch.Tensor:
        historical, forecast = self._encode_streams(history, forecast)
        history_tokens = historical.shape[1]
        fused = self.self_fusion(torch.cat((historical, forecast), dim=1))
        return self._predict(fused[:, history_tokens:])


class _CrossAttentionLayer(nn.Module):
    """Cross-attention and feed-forward residuals without target self-attention."""

    def __init__(self, config: TimestepTransformerConfig) -> None:
        super().__init__()
        embedding = config.decoder_embedding_dim
        dropout = config.decoder_dropout
        self.norm_first = config.decoder_norm_first
        self.attention = nn.MultiheadAttention(
            embedding, config.decoder_heads, dropout=dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(
            embedding, eps=config.decoder_layer_norm_eps
        )
        self.memory_norm = nn.LayerNorm(
            embedding, eps=config.decoder_layer_norm_eps
        )
        self.feedforward_norm = nn.LayerNorm(
            embedding, eps=config.decoder_layer_norm_eps
        )
        activation = (
            nn.GELU() if config.decoder_activation == "gelu" else nn.ReLU()
        )
        self.feedforward = nn.Sequential(
            nn.Linear(embedding, config.decoder_feedforward_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(config.decoder_feedforward_dim, embedding),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        if self.norm_first:
            attended = self._attend(self.query_norm(query), self.memory_norm(memory))
            query = query + attended
            feedforward = self.feedforward(self.feedforward_norm(query))
            return query + self.dropout(feedforward)
        query = self.query_norm(query + self._attend(query, memory))
        return self.feedforward_norm(query + self.dropout(self.feedforward(query)))

    def _attend(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.attention(
            query, memory, memory, need_weights=False
        )
        return self.dropout(attended)


__all__ = [
    "SeparateStreamCrossFusionTransformer",
    "SeparateStreamSelfFusionTransformer",
]
