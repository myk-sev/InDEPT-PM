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
MASKING_ALGORITHM_VERSION = "bulk-proposals-v1"
_BLOCK_PROPOSAL_CHUNK = 64


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
    _validate_inputs(values, observed, time_features)
    artificial = torch.zeros_like(observed)
    if stage in STAGES:
        _mask_curriculum(artificial, observed, stage, generator)
    else:
        _mask_tempo(artificial, observed, stage, generator)
    for item in range(len(values)):
        if artificial[item].any():
            continue
        known = observed[item].nonzero()
        if not len(known):
            raise ValueError("cannot mask a history with no observations")
        choice = known[torch.randint(len(known), (), generator=generator)]
        artificial[item, choice[0], choice[1]] = True
    return _build_masked_batch(values, observed, time_features, artificial)


def apply_mask(
    values: torch.Tensor,
    observed: torch.Tensor,
    time_features: torch.Tensor,
    target_mask: torch.Tensor,
) -> MaskedBatch:
    """Build model inputs from a validated, reusable artificial target mask."""
    _validate_inputs(values, observed, time_features)
    if target_mask.shape != observed.shape or target_mask.dtype != torch.bool:
        raise ValueError("target mask must be boolean and match values")
    if (target_mask & ~observed).any():
        raise ValueError("target mask may contain only observed values")
    return _build_masked_batch(values, observed, time_features, target_mask)


def _validate_inputs(
    values: torch.Tensor,
    observed: torch.Tensor,
    time_features: torch.Tensor,
) -> None:
    if values.shape != observed.shape or values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("values and observed must be [batch, time, 2]")
    if time_features.ndim != 3 or time_features.shape[:2] != values.shape[:2]:
        raise ValueError("time features must match the batch and history dimensions")


def _build_masked_batch(
    values: torch.Tensor,
    observed: torch.Tensor,
    time_features: torch.Tensor,
    target_mask: torch.Tensor,
) -> MaskedBatch:
    visible = observed & ~target_mask
    inputs = values.masked_fill(~visible, MASK_SENTINEL)
    features = torch.cat((inputs, time_features), dim=-1)
    return MaskedBatch(features, values, target_mask)


def _mask_tempo(
    masked: torch.Tensor,
    observed: torch.Tensor,
    stage: str,
    generator: torch.Generator,
) -> None:
    for item in range(len(masked)):
        _add_channel_points(masked[item], observed[item], 1, 0.15, generator)
    _add_channel_blocks(
        masked,
        observed,
        0,
        TEMPO_BRIDGE_MISSING_FRACTIONS[stage],
        (6, 12, 24, 48),
        generator,
    )


def _mask_curriculum(
    masked: torch.Tensor,
    observed: torch.Tensor,
    stage: str,
    generator: torch.Generator,
) -> None:
    fraction = 0.15 if stage == "points" else 0.10
    for item in range(len(masked)):
        _add_points(masked[item], observed[item], fraction, generator)
    if stage == "points":
        return
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
    start = max(0, observed.shape[1] - suffix)
    masked[:, start:, 0] = False
    masked[:, start:, 1] |= observed[:, start:, 1]


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
    batch_size, time, _ = masked.shape
    target = torch.ceil(observed[:, :, channel].sum(1) * fraction).to(torch.int64)
    attempts = torch.zeros(batch_size, dtype=torch.int64)
    length_values = torch.tensor(lengths, dtype=torch.int64)
    while True:
        active = (
            (masked[:, :, channel].sum(1) < target) & (attempts < time * 4)
        ).nonzero().flatten()
        if not len(active):
            break
        remaining = (time * 4 - attempts[active]).clamp(max=_BLOCK_PROPOSAL_CHUNK)
        intervals = _proposal_intervals(time, remaining, length_values, generator)
        current = masked[active, :, channel]
        cumulative = torch.cumsum(
            observed[active, None, :, channel] & intervals, 1, dtype=torch.int16
        ) > 0
        combined = cumulative | current[:, None]
        reached = combined.sum(2) >= target[active, None]
        chosen = _first_reached(reached, remaining)
        rows = torch.arange(len(active))
        previous = combined[rows, (chosen - 1).clamp(min=0)]
        previous = torch.where((chosen == 0)[:, None], current, previous)
        final_new = (
            observed[active, :, channel]
            & intervals[rows, chosen]
            & ~previous
        )
        needed = target[active] - previous.sum(1)
        final = previous | (
            final_new & (torch.cumsum(final_new, 1) <= needed[:, None])
        )
        selected = torch.where(
            reached.any(1)[:, None], final, combined[rows, chosen]
        )
        masked[active, :, channel] = selected
        attempts[active] += chosen + 1
    for item in range(batch_size):
        _add_channel_points(masked[item], observed[item], channel, fraction, generator)


def _add_blocks(
    masked: torch.Tensor,
    observed: torch.Tensor,
    fraction: float,
    lengths: tuple[int, ...],
    generator: torch.Generator,
) -> None:
    batch_size, time, channels = masked.shape
    target = torch.ceil(observed.sum((1, 2)) * fraction).to(torch.int64)
    attempts = torch.zeros(batch_size, dtype=torch.int64)
    length_values = torch.tensor(lengths, dtype=torch.int64)
    channel_values = torch.arange(channels)
    while True:
        active = ((masked.sum((1, 2)) < target) & (attempts < time * 4)).nonzero().flatten()
        if not len(active):
            return
        remaining = (time * 4 - attempts[active]).clamp(max=_BLOCK_PROPOSAL_CHUNK)
        intervals = _proposal_intervals(time, remaining, length_values, generator)
        width = intervals.shape[1]
        selected_channels = torch.randint(
            channels, (len(active), width), generator=generator
        )
        proposals = (
            observed[active, None]
            & intervals[..., None]
            & (channel_values[None, None, None] == selected_channels[..., None, None])
        )
        cumulative = torch.cumsum(proposals, 1, dtype=torch.int16) > 0
        combined = cumulative | masked[active, None]
        _select_proposals(masked, active, attempts, remaining, combined, target)


def _add_cross_channel_blocks(
    masked: torch.Tensor,
    observed: torch.Tensor,
    fraction: float,
    generator: torch.Generator,
) -> None:
    batch_size, time, _ = masked.shape
    target = torch.ceil(observed.sum((1, 2)) * fraction).to(torch.int64)
    attempts = torch.zeros(batch_size, dtype=torch.int64)
    length_values = torch.tensor((3, 6, 12), dtype=torch.int64)
    while True:
        active = ((masked.sum((1, 2)) < target) & (attempts < time * 4)).nonzero().flatten()
        if not len(active):
            return
        remaining = (time * 4 - attempts[active]).clamp(max=_BLOCK_PROPOSAL_CHUNK)
        covered = torch.cumsum(
            _proposal_intervals(time, remaining, length_values, generator),
            1,
            dtype=torch.int16,
        ) > 0
        initial = masked[active]
        outdoor = initial[:, None, :, 0] & ~covered
        indoor = initial[:, None, :, 1] | (observed[active, None, :, 1] & covered)
        counts = outdoor.sum(2) + indoor.sum(2)
        reached = counts >= target[active, None]
        chosen = _first_reached(reached, remaining)
        rows = torch.arange(len(active))
        masked[active, :, 0] = outdoor[rows, chosen]
        masked[active, :, 1] = indoor[rows, chosen]
        attempts[active] += chosen + 1


def _proposal_intervals(
    time: int,
    remaining: torch.Tensor,
    length_values: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    width = int(remaining.max().item())
    lengths = length_values[
        torch.randint(len(length_values), (len(remaining), width), generator=generator)
    ].clamp(max=time)
    starts = torch.empty_like(lengths)
    for length in lengths.unique().tolist():
        selected = lengths == length
        starts[selected] = torch.randint(
            time - length + 1, (int(selected.sum().item()),), generator=generator
        )
    positions = torch.arange(time)
    valid = torch.arange(width)[None] < remaining[:, None]
    return (
        (positions[None, None] >= starts[..., None])
        & (positions[None, None] < (starts + lengths)[..., None])
        & valid[..., None]
    )


def _select_proposals(
    masked: torch.Tensor,
    active: torch.Tensor,
    attempts: torch.Tensor,
    remaining: torch.Tensor,
    proposals: torch.Tensor,
    target: torch.Tensor,
) -> None:
    reached = proposals.sum((2, 3)) >= target[active, None]
    chosen = _first_reached(reached, remaining)
    masked[active] = proposals[torch.arange(len(active)), chosen]
    attempts[active] += chosen + 1


def _first_reached(reached: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
    first = reached.to(torch.int8).argmax(1)
    return torch.where(reached.any(1), first, remaining - 1)
