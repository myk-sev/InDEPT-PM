from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data_loader import DualEncoderDataset, DualEncoderLoaders, create_data_loaders


CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class ModelConfig:
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
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for stream in ("history", "forecast"):
            if getattr(self, f"{stream}_patch_stride") > getattr(
                self, f"{stream}_patch_size"
            ):
                raise ValueError(
                    f"{stream}_patch_stride cannot exceed {stream}_patch_size"
                )

        for stream in ("history", "forecast", "decoder"):
            embedding = getattr(self, f"{stream}_embedding_dim")
            heads = getattr(self, f"{stream}_heads")
            head_dim = getattr(self, f"{stream}_head_dim")
            if embedding < 1 or heads < 1 or head_dim < 1:
                raise ValueError(
                    f"{stream} embedding, heads, and head dimension must be positive"
                )
            if embedding != heads * head_dim:
                raise ValueError(
                    f"{stream}_embedding_dim ({embedding}) must equal "
                    f"{stream}_heads ({heads}) x {stream}_head_dim ({head_dim})"
                )
            dropout = getattr(self, f"{stream}_dropout")
            if not 0 <= dropout < 1:
                raise ValueError(f"{stream}_dropout must be in [0, 1)")
            if getattr(self, f"{stream}_activation") not in {"relu", "gelu"}:
                raise ValueError(f"{stream}_activation must be relu or gelu")
            if getattr(self, f"{stream}_layer_norm_eps") <= 0:
                raise ValueError(f"{stream}_layer_norm_eps must be positive")


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
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.history_patches = PatchEmbedding(
            config.history_hours,
            8,
            config.history_patch_size,
            config.history_patch_stride,
            config.history_embedding_dim,
        )
        self.forecast_patches = PatchEmbedding(
            config.prediction_hours,
            1,
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
    if history.ndim != 3 or history.shape[2] != 6:
        raise ValueError(
            "history must contain outdoor PM2.5, indoor PM2.5, and four time features"
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


def _encoder(config: ModelConfig, stream: str) -> nn.TransformerEncoder:
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


@dataclass(frozen=True)
class ZScores:
    indoor_mean: float
    indoor_std: float
    history_outdoor_mean: float
    history_outdoor_std: float
    forecast_mean: float
    forecast_std: float

    def denormalize_indoor(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.indoor_std + self.indoor_mean


def fit_zscores(batches: Iterable[dict[str, torch.Tensor]]) -> ZScores:
    indoor = [0.0, 0.0, 0]
    history_outdoor = [0.0, 0.0, 0]
    forecast = [0.0, 0.0, 0]
    for batch in batches:
        _accumulate(indoor, batch["history"][..., 1])
        _accumulate(indoor, batch["target"])
        _accumulate(history_outdoor, batch["history"][..., 0])
        _accumulate(forecast, batch["forecast"])
    indoor_mean, indoor_std = _mean_std(indoor, "indoor")
    history_mean, history_std = _mean_std(history_outdoor, "historical outdoor")
    forecast_mean, forecast_std = _mean_std(forecast, "forecast")
    return ZScores(
        indoor_mean,
        indoor_std,
        history_mean,
        history_std,
        forecast_mean,
        forecast_std,
    )


def _accumulate(total: list[float | int], values: torch.Tensor) -> None:
    values = values.detach().double()
    values = values[torch.isfinite(values)]
    total[0] += values.sum().item()
    total[1] += values.square().sum().item()
    total[2] += values.numel()


def _mean_std(total: list[float | int], label: str) -> tuple[float, float]:
    count = int(total[2])
    if not count:
        raise ValueError(f"cannot fit {label} z-score without training values")
    mean = float(total[0]) / count
    variance = float(total[1]) / count - mean * mean
    if variance <= 0:
        raise ValueError(f"{label} training values must have non-zero variance")
    return mean, math.sqrt(variance)


def normalize_batch(
    batch: dict[str, torch.Tensor],
    zscores: ZScores,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history = batch["history"].to(device, non_blocking=True).clone()
    forecast = batch["forecast"].to(device, non_blocking=True).clone()
    target = batch["target"].to(device, non_blocking=True).clone()
    history[..., 0].sub_(zscores.history_outdoor_mean).div_(
        zscores.history_outdoor_std
    )
    history[..., 1].sub_(zscores.indoor_mean).div_(zscores.indoor_std)
    forecast.sub_(zscores.forecast_mean).div_(zscores.forecast_std)
    target.sub_(zscores.indoor_mean).div_(zscores.indoor_std)
    return history, forecast, target


def make_loss(name: str, huber_delta: float = 1.0) -> nn.Module:
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        return nn.HuberLoss(delta=huber_delta)
    raise ValueError("loss must be mae, mse, or huber")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    zscores: ZScores,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_total = absolute_total = squared_total = 0.0
    count = 0

    for batch in loader:
        history, forecast, target = normalize_batch(batch, zscores, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(history, forecast)
            loss = loss_function(prediction, target)
        if training:
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        physical_prediction = zscores.denormalize_indoor(prediction.detach())
        physical_target = zscores.denormalize_indoor(target.detach())
        error = physical_prediction - physical_target
        elements = target.numel()
        loss_total += loss.item() * elements
        absolute_total += error.abs().sum().item()
        squared_total += error.square().sum().item()
        count += elements

    if not count:
        raise ValueError("data loader contains no samples")
    mse = squared_total / count
    return {
        "loss": loss_total / count,
        "mae": absolute_total / count,
        "mse": mse,
        "rmse": math.sqrt(mse),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: ModelConfig,
    zscores: ZScores,
    training_config: dict,
    epoch: int,
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": asdict(config),
            "normalization": asdict(zscores),
            "training_config": training_config,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state": model.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[DualEncoderPatchTransformer, ModelConfig, ZScores, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "checkpoint predates the missing-aware history architecture; retrain it"
        )
    config = ModelConfig(**checkpoint["model_config"])
    zscores = ZScores(**checkpoint["normalization"])
    model = DualEncoderPatchTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, config, zscores, checkpoint


def plot_prediction(
    output: Path,
    sample: dict[str, torch.Tensor],
    prediction: torch.Tensor,
    config: ModelConfig,
    location_id: str,
    split: str,
    sample_index: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    import seaborn as sns

    history_hours = np.arange(-config.history_hours + 1, 1)
    future_hours = np.arange(1, config.prediction_hours + 1)
    history = sample["history"].cpu().numpy()
    forecast = sample["forecast"].squeeze(1).cpu().numpy()
    target = sample["target"].cpu().numpy()
    predicted = prediction.detach().cpu().numpy()
    error = predicted - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    displayed = np.concatenate(
        (history[:, :2].ravel(), forecast, target, predicted)
    )
    displayed = displayed[np.isfinite(displayed)]
    padding = max(float(np.ptp(displayed)) * 0.08, 0.5)
    y_limits = (
        max(0, float(displayed.min()) - padding),
        float(displayed.max()) + padding,
    )
    history_title = (
        f"PAST {config.history_hours // 24} DAYS"
        if config.history_hours % 24 == 0
        else f"PAST {config.history_hours} HOURS"
    )
    anchor = datetime.fromtimestamp(
        int(sample["anchor_time_utc"]), timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    sns.set_theme(style="ticks")
    figure, (history_axis, forecast_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        sharey=True,
        gridspec_kw={"width_ratios": (1, 2), "wspace": 0.05},
    )
    history_lines = (
        (
            history_hours,
            history[:, 0],
            "Outdoor observed",
            "-",
            "#9C6B30",
        ),
        (
            history_hours,
            history[:, 1],
            "Indoor observed",
            "-",
            "#40566F",
        ),
    )
    forecast_lines = (
        (future_hours, forecast, "Outdoor forecast", "--", "#9C6B30"),
        (future_hours, target, "Indoor measured", "-", "#40566F"),
        (
            future_hours,
            predicted,
            "Model prediction",
            "--",
            "#267873",
        ),
    )
    for axis, lines in (
        (history_axis, history_lines),
        (forecast_axis, forecast_lines),
    ):
        for hours, values, label, style, color in lines:
            sns.lineplot(
                x=hours,
                y=values,
                label=label,
                linestyle=style,
                color=color,
                linewidth=(
                    2.25
                    if label in {"Indoor measured", "Model prediction"}
                    else 1.6
                ),
                ax=axis,
            )

    forecast_axis.axvline(
        0, color="#555A60", linestyle=":", linewidth=1.5
    )
    forecast_axis.annotate(
        "Forecast begins",
        xy=(0, y_limits[1]),
        xytext=(5, -4),
        textcoords="offset points",
        ha="left",
        va="top",
        color="#555A60",
        fontsize=9,
    )
    history_axis.set(
        xlabel="Days before forecast",
        ylabel="PM2.5 (µg/m³)",
        xlim=(-config.history_hours + 1, 0),
        ylim=y_limits,
    )
    forecast_axis.set(
        xlabel="Forecast lead hour",
        ylabel="",
        xlim=(0, config.prediction_hours),
    )
    history_axis.spines["right"].set_visible(False)
    forecast_axis.spines["left"].set_visible(False)
    forecast_axis.tick_params(axis="y", left=False)
    history_axis.set_title(history_title, loc="left", fontsize=10, color="#555A60")
    forecast_axis.set_title(
        f"{config.prediction_hours}-HOUR HOLDOUT EVALUATION",
        loc="left",
        fontsize=10,
        color="#555A60",
    )
    if config.history_hours == 168:
        history_axis.set_xticks(
            (-168, -120, -72, -24, 0),
            ("−7 d", "−5 d", "−3 d", "−1 d", "0"),
        )
    forecast_axis.set_xticks(
        np.arange(0, config.prediction_hours + 1, 6)
    )
    for axis in (history_axis, forecast_axis):
        axis.grid(axis="y", color="#DFE3E7", linewidth=0.8)
        axis.grid(axis="x", visible=False)

    handles = []
    labels = []
    for axis in (history_axis, forecast_axis):
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
        axis.get_legend().remove()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        f"Model error averaged {mae:.2f} µg/m³ over {config.prediction_hours} hours",
        y=0.99,
        fontsize=16,
    )
    figure.text(
        0.5,
        0.925,
        f"{location_id} · {anchor} · retrospective {split} sample {sample_index} · RMSE {rmse:.2f} µg/m³",
        ha="center",
        color="#555A60",
        fontsize=10,
    )
    figure.text(0.35, 0.14, "//", ha="center", color="#555A60", fontsize=13)
    figure.subplots_adjust(top=0.77, bottom=0.14, left=0.09, right=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.xpu.is_available():
            name = "xpu"
        else:
            name = "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "xpu" and not torch.xpu.is_available():
        raise ValueError("XPU was requested but is not available")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(
    pairs: Path,
    indoor_history: list[Path],
    outdoor_history: Path,
    forecast_root: Path,
    config: ModelConfig,
    training_config: dict,
    device: torch.device,
) -> tuple[DualEncoderDataset, DualEncoderLoaders]:
    dataset = DualEncoderDataset(
        pairs,
        indoor_history,
        outdoor_history,
        forecast_root,
        history_hours=config.history_hours,
        forecast_hours=config.prediction_hours,
        minimum_outdoor_history_hours=training_config.get(
            "minimum_outdoor_history_hours", 24
        ),
        maximum_outdoor_age_hours=training_config.get(
            "maximum_outdoor_age_hours", 48
        ),
    )
    loaders = create_data_loaders(
        dataset,
        batch_size=training_config["batch_size"],
        train_fraction=training_config["train_fraction"],
        validation_fraction=training_config["validation_fraction"],
        location_holdout_fraction=training_config["location_holdout_fraction"],
        seed=training_config["seed"],
        num_workers=training_config["num_workers"],
        pin_memory=device.type in {"cuda", "xpu"},
    )
    return dataset, loaders


def train(args: argparse.Namespace) -> None:
    config = ModelConfig(
        **{field.name: getattr(args, field.name) for field in fields(ModelConfig)}
    )
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if args.gradient_clip < 0 or args.early_stopping_patience < 0:
        raise ValueError(
            "gradient_clip and early_stopping_patience cannot be negative"
        )

    set_seed(args.seed)
    device = resolve_device(args.device)
    training_config = {
        "pairs": str(args.pairs.resolve()),
        "indoor_history": [str(path.resolve()) for path in args.indoor_history],
        "outdoor_history": str(args.outdoor_history.resolve()),
        "forecast_root": str(args.forecast_root.resolve()),
        "batch_size": args.batch_size,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "location_holdout_fraction": args.location_holdout_fraction,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "loss": args.loss,
        "huber_delta": args.huber_delta,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "early_stopping_patience": args.early_stopping_patience,
        "epochs": args.epochs,
        "device": args.device,
        "minimum_outdoor_history_hours": args.minimum_outdoor_history_hours,
        "maximum_outdoor_age_hours": args.maximum_outdoor_age_hours,
    }
    _, loaders = build_loaders(
        args.pairs,
        args.indoor_history,
        args.outdoor_history,
        args.forecast_root,
        config,
        training_config,
        device,
    )
    zscores = fit_zscores(loaders.train)
    model = DualEncoderPatchTransformer(config).to(device)
    loss_function = make_loss(args.loss, args.huber_delta)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    counts = {
        name: len(getattr(loaders, name).dataset)
        for name in ("train", "validation", "temporal_test", "location_test")
    }
    print(f"device={device} samples={counts}")
    print(
        f"indoor z-score mean={zscores.indoor_mean:.6g} "
        f"std={zscores.indoor_std:.6g}; "
        f"historical outdoor mean={zscores.history_outdoor_mean:.6g} "
        f"std={zscores.history_outdoor_std:.6g}; "
        f"forecast mean={zscores.forecast_mean:.6g} "
        f"std={zscores.forecast_std:.6g}"
    )

    best_loss = math.inf
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(
            model,
            loaders.train,
            loss_function,
            zscores,
            device,
            optimizer,
            args.gradient_clip,
        )
        validation = run_epoch(
            model, loaders.validation, loss_function, zscores, device
        )
        elapsed = time.perf_counter() - started
        print(
            f"epoch={epoch}/{args.epochs} elapsed={elapsed:.1f}s "
            f"train_loss={training['loss']:.6g} "
            f"val_loss={validation['loss']:.6g} "
            f"train_mae={training['mae']:.6g} "
            f"train_mse={training['mse']:.6g} "
            f"train_rmse={training['rmse']:.6g} "
            f"val_mae={validation['mae']:.6g} "
            f"val_mse={validation['mse']:.6g} "
            f"val_rmse={validation['rmse']:.6g}"
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            stale_epochs = 0
            save_checkpoint(
                args.checkpoint,
                model,
                config,
                zscores,
                training_config,
                epoch,
                best_loss,
            )
        else:
            stale_epochs += 1
            if (
                args.early_stopping_patience
                and stale_epochs >= args.early_stopping_patience
            ):
                print(f"early stopping after {epoch} epochs")
                break
    print(f"best_{args.loss}={best_loss:.6g} checkpoint={args.checkpoint}")


def infer(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, config, zscores, checkpoint = load_checkpoint(args.checkpoint, device)
    training_config = dict(checkpoint["training_config"])
    training_config["batch_size"] = 1
    training_config["num_workers"] = 0
    dataset, loaders = build_loaders(
        args.pairs,
        args.indoor_history,
        args.outdoor_history,
        args.forecast_root,
        config,
        training_config,
        device,
    )
    loader = {
        "train": loaders.train,
        "validation": loaders.validation,
        "temporal-test": loaders.temporal_test,
        "location-test": loaders.location_test,
    }[args.split]
    if not 0 <= args.sample_index < len(loader.dataset):
        raise IndexError(
            f"sample_index must be between 0 and {len(loader.dataset) - 1}"
        )
    sample = loader.dataset[args.sample_index]
    batch = {
        name: sample[name].unsqueeze(0)
        for name in ("history", "forecast", "target")
    }
    history, forecast, _ = normalize_batch(batch, zscores, device)
    model.eval()
    with torch.no_grad():
        prediction = zscores.denormalize_indoor(model(history, forecast))[0]
    location_index = int(sample["location_index"])
    plot_prediction(
        args.output,
        sample,
        prediction,
        config,
        dataset.location_ids[location_index],
        args.split,
        args.sample_index,
    )
    print(f"saved diagnostic graph: {args.output}")

def _add_stream_arguments(
    parser: argparse.ArgumentParser, stream: str, patches: bool
) -> None:
    defaults = ModelConfig()
    prefix = stream.replace("_", "-")
    if patches:
        parser.add_argument(
            f"--{prefix}-patch-size",
            type=int,
            default=getattr(defaults, f"{stream}_patch_size"),
        )
        parser.add_argument(
            f"--{prefix}-patch-stride",
            type=int,
            default=getattr(defaults, f"{stream}_patch_stride"),
        )
    parser.add_argument(
        f"--{prefix}-embedding-dim",
        type=int,
        default=getattr(defaults, f"{stream}_embedding_dim"),
    )
    parser.add_argument(
        f"--{prefix}-heads",
        type=int,
        default=getattr(defaults, f"{stream}_heads"),
    )
    parser.add_argument(
        f"--{prefix}-head-dim",
        type=int,
        default=getattr(defaults, f"{stream}_head_dim"),
    )
    parser.add_argument(
        f"--{prefix}-layers",
        type=int,
        default=getattr(defaults, f"{stream}_layers"),
    )
    parser.add_argument(
        f"--{prefix}-feedforward-dim",
        type=int,
        default=getattr(defaults, f"{stream}_feedforward_dim"),
    )
    parser.add_argument(
        f"--{prefix}-dropout",
        type=float,
        default=getattr(defaults, f"{stream}_dropout"),
    )
    parser.add_argument(
        f"--{prefix}-activation",
        choices=("relu", "gelu"),
        default=getattr(defaults, f"{stream}_activation"),
    )
    parser.add_argument(
        f"--{prefix}-norm-first",
        action=argparse.BooleanOptionalAction,
        default=getattr(defaults, f"{stream}_norm_first"),
    )
    parser.add_argument(
        f"--{prefix}-layer-norm-eps",
        type=float,
        default=getattr(defaults, f"{stream}_layer_norm_eps"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and inspect a dual-encoder PM2.5 patch transformer.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser(
        "train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument("--pairs", type=Path, required=True)
    train_parser.add_argument(
        "--indoor-history", type=Path, action="append", required=True
    )
    train_parser.add_argument("--outdoor-history", type=Path, required=True)
    train_parser.add_argument("--forecast-root", type=Path, required=True)
    train_parser.add_argument("--checkpoint", type=Path, required=True)
    train_parser.add_argument("--history-hours", type=int, default=168)
    train_parser.add_argument("--prediction-hours", type=int, default=36)
    train_parser.add_argument("--minimum-outdoor-history-hours", type=int, default=24)
    train_parser.add_argument("--maximum-outdoor-age-hours", type=int, default=48)
    _add_stream_arguments(train_parser, "history", True)
    _add_stream_arguments(train_parser, "forecast", True)
    _add_stream_arguments(train_parser, "decoder", False)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_parser.add_argument("--loss", choices=("mae", "mse", "huber"), default="mse")
    train_parser.add_argument("--huber-delta", type=float, default=1.0)
    train_parser.add_argument("--train-fraction", type=float, default=0.70)
    train_parser.add_argument("--validation-fraction", type=float, default=0.15)
    train_parser.add_argument(
        "--location-holdout-fraction", type=float, default=0.20
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--early-stopping-patience", type=int, default=10)
    train_parser.set_defaults(function=train)

    infer_parser = commands.add_parser(
        "infer", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    infer_parser.add_argument("--pairs", type=Path, required=True)
    infer_parser.add_argument(
        "--indoor-history", type=Path, action="append", required=True
    )
    infer_parser.add_argument("--outdoor-history", type=Path, required=True)
    infer_parser.add_argument("--forecast-root", type=Path, required=True)
    infer_parser.add_argument("--checkpoint", type=Path, required=True)
    infer_parser.add_argument(
        "--split",
        choices=("train", "validation", "temporal-test", "location-test"),
        default="temporal-test",
    )
    infer_parser.add_argument("--sample-index", type=int, default=0)
    infer_parser.add_argument(
        "--output", type=Path, default=Path("pm25_inference.png")
    )
    infer_parser.add_argument("--device", default="auto")
    infer_parser.set_defaults(function=infer)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
