from __future__ import annotations

import math
from dataclasses import dataclass

import torch


STAGES = (
    "points",
    "short_blocks",
    "mixed_blocks",
    "cross_channel",
    "suffix_3",
    "suffix_6",
    "suffix_12",
)
TEMPO_BRIDGE_MISSING_FRACTIONS = {
    "tempo_bridge_50": 0.50,
    "tempo_bridge_70": 0.70,
    "tempo_bridge_86": 6 / 7,
}
TEMPO_BRIDGE_STAGES = tuple(TEMPO_BRIDGE_MISSING_FRACTIONS)
ALL_STAGES = STAGES + TEMPO_BRIDGE_STAGES
MASK_SENTINEL = -9.0


@dataclass(frozen=True)
class MaskedBatch:
    features: torch.Tensor
    target: torch.Tensor
    target_mask: torch.Tensor


def mask_batch(
    values: torch.Tensor,
    observed: torch.Tensor,
    time_features: torch.Tensor,
    stage: str,
    generator: torch.Generator,
) -> MaskedBatch:
    """Replace missing values with a sentinel and target only artificial masks."""
    if stage not in ALL_STAGES:
        raise ValueError(f"unknown masking stage: {stage}")
    if values.shape != observed.shape or values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("values and observed must be [batch, time, 2]")
    if time_features.shape[:2] != values.shape[:2]:
        raise ValueError("time features must match the batch and history dimensions")
    artificial = torch.zeros_like(observed)
    for item in range(len(values)):
        _mask_item(artificial[item], observed[item], stage, generator)
        if not artificial[item].any():
            known = observed[item].nonzero()
            if not len(known):
                raise ValueError("cannot mask a history with no observations")
            choice = known[torch.randint(len(known), (), generator=generator)]
            artificial[item, choice[0], choice[1]] = True
    visible = observed & ~artificial
    inputs = values.masked_fill(~visible, MASK_SENTINEL)
    features = torch.cat((inputs, time_features), dim=-1)
    return MaskedBatch(features, values, artificial)


def _mask_item(
    masked: torch.Tensor,
    observed: torch.Tensor,
    stage: str,
    generator: torch.Generator,
) -> None:
    if stage in TEMPO_BRIDGE_MISSING_FRACTIONS:
        _add_channel_points(masked, observed, 1, 0.15, generator)
        _add_channel_blocks(
            masked,
            observed,
            0,
            TEMPO_BRIDGE_MISSING_FRACTIONS[stage],
            (6, 12, 24, 48),
            generator,
        )
        return
    if stage == "points":
        _add_points(masked, observed, 0.15, generator)
        return
    _add_points(masked, observed, 0.10, generator)
    if stage == "short_blocks":
        _add_blocks(masked, observed, 0.25, (2, 3), generator)
        return
    _add_blocks(masked, observed, 0.25, (1, 3, 6), generator)
    if stage == "mixed_blocks":
        _add_blocks(masked, observed, 0.35, (1, 3, 6), generator)
        return
    if stage == "cross_channel":
        _add_cross_channel_blocks(masked, observed, 0.35, generator)
        return
    suffix = int(stage.rsplit("_", 1)[1])
    _add_blocks(masked, observed, 0.30, (1, 3, 6), generator)
    start = max(0, len(observed) - suffix)
    masked[start:, 0] = False
    masked[start:, 1] |= observed[start:, 1]


def _add_points(
    masked: torch.Tensor,
    observed: torch.Tensor,
    fraction: float,
    generator: torch.Generator,
) -> None:
    known = (observed & ~masked).nonzero()
    target = max(1, math.ceil(observed.sum().item() * fraction))
    count = min(len(known), max(0, target - int(masked.sum())))
    if count:
        chosen = known[torch.randperm(len(known), generator=generator)[:count]]
        masked[chosen[:, 0], chosen[:, 1]] = True


def _add_channel_points(
    masked: torch.Tensor,
    observed: torch.Tensor,
    channel: int,
    fraction: float,
    generator: torch.Generator,
) -> None:
    known = (observed[:, channel] & ~masked[:, channel]).nonzero().flatten()
    target = math.ceil(observed[:, channel].sum().item() * fraction)
    count = min(len(known), max(0, target - int(masked[:, channel].sum())))
    if count:
        chosen = known[torch.randperm(len(known), generator=generator)[:count]]
        masked[chosen, channel] = True


def _add_channel_blocks(
    masked: torch.Tensor,
    observed: torch.Tensor,
    channel: int,
    fraction: float,
    lengths: tuple[int, ...],
    generator: torch.Generator,
) -> None:
    target = math.ceil(observed[:, channel].sum().item() * fraction)
    attempts = 0
    while masked[:, channel].sum().item() < target and attempts < len(observed) * 4:
        length = lengths[torch.randint(len(lengths), (), generator=generator).item()]
        length = min(length, len(observed))
        start = torch.randint(len(observed) - length + 1, (), generator=generator).item()
        known = (
            observed[start : start + length, channel]
            & ~masked[start : start + length, channel]
        ).nonzero().flatten()
        remaining = target - int(masked[:, channel].sum())
        masked[start + known[:remaining], channel] = True
        attempts += 1
    _add_channel_points(masked, observed, channel, fraction, generator)


def _add_blocks(
    masked: torch.Tensor,
    observed: torch.Tensor,
    fraction: float,
    lengths: tuple[int, ...],
    generator: torch.Generator,
) -> None:
    target = math.ceil(observed.sum().item() * fraction)
    attempts = 0
    while masked.sum().item() < target and attempts < len(observed) * 4:
        length = lengths[torch.randint(len(lengths), (), generator=generator).item()]
        start = torch.randint(max(1, len(observed) - length + 1), (), generator=generator).item()
        channel = torch.randint(2, (), generator=generator).item()
        masked[start : start + length, channel] |= observed[start : start + length, channel]
        attempts += 1


def _add_cross_channel_blocks(
    masked: torch.Tensor,
    observed: torch.Tensor,
    fraction: float,
    generator: torch.Generator,
) -> None:
    target = math.ceil(observed.sum().item() * fraction)
    attempts = 0
    while masked.sum().item() < target and attempts < len(observed) * 4:
        length = (3, 6, 12)[torch.randint(3, (), generator=generator).item()]
        start = torch.randint(max(1, len(observed) - length + 1), (), generator=generator).item()
        masked[start : start + length, 0] = False
        masked[start : start + length, 1] |= observed[start : start + length, 1]
        attempts += 1
