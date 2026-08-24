"""Classify post-exclusion indoor/outdoor PurpleAir pair responsiveness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pair_movement_correlation.analyze_movement_correlation import (
    pearson,
    ranks,
    winsorize,
)
from purpleair_pair_exclusions.detect_pair_exclusions import read_histories
from purpleair_pair_exclusions.outdoor_quality import (
    exclude_outdoor_readings,
    read_outdoor_exclusions,
)


HOUR = 3600
ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = ROOT / "inputs"
DATA_ROOT = ROOT / "data"
MASKED_INPUTS = INPUT_ROOT / "masked_pretraining"
DEFAULT_PAIRS = ROOT / "purpleair_pair_exclusions" / "results" / "selected_pairs.csv"
DEFAULT_INDOOR = (DATA_ROOT / "purple air" / "all_indoor_pm25.csv",)
DEFAULT_OUTDOOR = (DATA_ROOT / "purple air" / "all_outdoor_pm25.csv",)
DEFAULT_OUTDOOR_EXCLUSIONS = DATA_ROOT / "exclusions" / "excluded_outdoor_purpleair_ranges.csv"
DEFAULT_OUTPUT = MASKED_INPUTS / "responsiveness"
PAIR_FIELDS = (
    "indoor_sensor_id",
    "indoor_name",
    "outdoor_sensor_id",
    "outdoor_name",
    "distance_meters",
    "cohort_sources",
)
METRIC_FIELDS = (
    "pair_id",
    *PAIR_FIELDS,
    "indoor_hours",
    "outdoor_hours",
    "overlap_hours",
    "movement_hours",
    "best_lag_hours",
    "best_lag_pearson_r",
    "response_gain",
    "direction_agreement",
    "early_pearson_r",
    "late_pearson_r",
    "stable_pearson_r",
    "classification_confidence",
    "responsiveness_score",
    "responsiveness_percentile",
    "responsiveness_tier",
    "curriculum_stage",
    "status",
)
WEIGHTS = {
    "best_lag_pearson_r": 0.45,
    "response_gain": 0.20,
    "direction_agreement": 0.15,
    "stable_pearson_r": 0.10,
    "response_speed": 0.10,
}


def read_pairs(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(PAIR_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing pair columns: {', '.join(sorted(missing))}")
        rows, seen = [], set()
        for number, row in enumerate(reader, 2):
            try:
                indoor = int(row["indoor_sensor_id"])
                outdoor = int(row["outdoor_sensor_id"])
                distance = float(row["distance_meters"])
                key = indoor, outdoor
                if min(key) < 1 or key in seen or distance < 0 or not math.isfinite(distance):
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid or duplicate pair row {number}") from error
            seen.add(key)
            rows.append(
                {
                    "indoor_sensor_id": indoor,
                    "indoor_name": row["indoor_name"],
                    "outdoor_sensor_id": outdoor,
                    "outdoor_name": row["outdoor_name"],
                    "distance_meters": distance,
                    "cohort_sources": row["cohort_sources"],
                }
            )
    if not rows:
        raise ValueError("pair CSV contains no rows")
    return rows


def movement_samples(
    indoor: dict[int, float], outdoor: dict[int, float], lag_hours: int
) -> tuple[np.ndarray, np.ndarray]:
    lag = lag_hours * HOUR
    samples = [
        (
            outdoor[timestamp] - outdoor[timestamp - HOUR],
            indoor[timestamp + lag] - indoor[timestamp + lag - HOUR],
        )
        for timestamp in sorted(outdoor)
        if timestamp - HOUR in outdoor
        and timestamp + lag in indoor
        and timestamp + lag - HOUR in indoor
    ]
    if not samples:
        return np.asarray([]), np.asarray([])
    outside, inside = zip(*samples, strict=True)
    return np.asarray(outside), np.asarray(inside)


def _correlation(outside: np.ndarray, inside: np.ndarray, percent: float) -> float | None:
    if len(outside) < 2:
        return None
    return pearson(winsorize(outside, percent), winsorize(inside, percent))


def _gain(outside: np.ndarray, inside: np.ndarray, percent: float) -> float | None:
    outside, inside = winsorize(outside, percent), winsorize(inside, percent)
    centered = outside - outside.mean()
    denominator = float(centered @ centered)
    if not denominator:
        return None
    return float(centered @ (inside - inside.mean()) / denominator)


def pair_metrics(
    pair: dict[str, object],
    indoor: dict[int, float],
    outdoor: dict[int, float],
    minimum_movements: int,
    maximum_lag: int,
    winsor_percent: float,
) -> dict[str, object]:
    row = {
        "pair_id": f"{pair['indoor_sensor_id']}-{pair['outdoor_sensor_id']}",
        **pair,
        "indoor_hours": len(indoor),
        "outdoor_hours": len(outdoor),
        "overlap_hours": len(indoor.keys() & outdoor.keys()),
        "responsiveness_tier": "unclassified",
    }
    if not indoor or not outdoor:
        row["status"] = (
            "missing_both_histories"
            if not indoor and not outdoor
            else "missing_indoor_history" if not indoor else "missing_outdoor_history"
        )
        return row

    candidates = []
    for lag in range(maximum_lag + 1):
        outside, inside = movement_samples(indoor, outdoor, lag)
        if len(outside) < minimum_movements:
            continue
        correlation = _correlation(outside, inside, winsor_percent)
        if correlation is not None:
            candidates.append((correlation, -lag, outside, inside))
    if not candidates:
        outside, _ = movement_samples(indoor, outdoor, 0)
        row.update(movement_hours=len(outside), status="insufficient_movement_hours")
        return row

    correlation, negative_lag, outside, inside = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    lag = -negative_lag
    moving = (outside != 0) & (inside != 0)
    midpoint = len(outside) // 2
    early = _correlation(outside[:midpoint], inside[:midpoint], winsor_percent)
    late = _correlation(outside[midpoint:], inside[midpoint:], winsor_percent)
    stable = min(early, late) if early is not None and late is not None else None
    row.update(
        movement_hours=len(outside),
        best_lag_hours=lag,
        best_lag_pearson_r=correlation,
        response_gain=_gain(outside, inside, winsor_percent),
        direction_agreement=(
            float(np.mean(np.sign(outside[moving]) == np.sign(inside[moving])))
            if moving.any()
            else None
        ),
        early_pearson_r=early,
        late_pearson_r=late,
        stable_pearson_r=stable,
        classification_confidence=(
            "high"
            if len(outside) >= 1_000 and stable is not None
            else "moderate" if len(outside) >= 336 and stable is not None else "low"
        ),
        status="analyzed",
    )
    return row


def _percentiles(rows: list[dict[str, object]], field: str) -> dict[str, float]:
    if field == "response_speed":
        values = -np.asarray([float(row["best_lag_hours"]) for row in rows])
        ranked = ranks(values) / max(len(rows) - 1, 1)
        return {
            str(row["pair_id"]): float(rank)
            for row, rank in zip(rows, ranked, strict=True)
        }
    valid = [row for row in rows if row.get(field) is not None]
    values = np.asarray([max(float(row[field]), 0.0) for row in valid])
    ranked = ranks(values) / max(len(valid) - 1, 1)
    result = {str(row["pair_id"]): 0.5 for row in rows}
    result.update(
        {
            str(row["pair_id"]): float(rank)
            for row, rank in zip(valid, ranked, strict=True)
        }
    )
    return result


def classify(rows: list[dict[str, object]]) -> None:
    analyzed = [row for row in rows if row["status"] == "analyzed"]
    if not analyzed:
        return
    percentiles = {field: _percentiles(analyzed, field) for field in WEIGHTS}
    for row in analyzed:
        pair_id = str(row["pair_id"])
        row["responsiveness_score"] = 100 * sum(
            weight * percentiles[field][pair_id] for field, weight in WEIGHTS.items()
        )
    ordered = sorted(
        analyzed,
        key=lambda row: (-float(row["responsiveness_score"]), str(row["pair_id"])),
    )
    count = len(ordered)
    for index, row in enumerate(ordered):
        row["responsiveness_percentile"] = (
            100.0 if count == 1 else 100 * (count - index - 1) / (count - 1)
        )
        if index < math.ceil(count / 3):
            row.update(responsiveness_tier="high", curriculum_stage=1)
        elif index < math.ceil(2 * count / 3):
            row.update(responsiveness_tier="moderate", curriculum_stage=2)
        else:
            row.update(responsiveness_tier="low", curriculum_stage=3)


def source_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {"path": str(path.resolve())}
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        record.update(bytes=path.stat().st_size, sha256=digest.hexdigest())
    return record


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, METRIC_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["indoor_sensor_id"])))


def write_report(path: Path, summary: dict[str, object]) -> None:
    counts = summary["counts"]
    status_rows = [
        f"| {status} | {count} |"
        for status, count in summary["status_counts"].items()
    ]
    lines = [
        "# Indoor/outdoor pair responsiveness classification",
        "",
        "## Result",
        "",
        f"All {counts['pairs']} post-exclusion training pairs were evaluated. "
        f"{counts['analyzed']} had enough paired movement data to classify; "
        f"{counts['unclassified']} remain `unclassified` rather than being guessed.",
        "",
        "| Tier | Pairs | Curriculum |",
        "|---|---:|---|",
        f"| high | {counts['high']} | Stage 1 |",
        f"| moderate | {counts['moderate']} | Add at Stage 2 |",
        f"| low | {counts['low']} | Add at Stage 3 |",
        f"| unclassified | {counts['unclassified']} | Hold until history is sufficient |",
        "",
        "| Analysis status | Pairs |",
        "|---|---:|",
        *status_rows,
        "",
        "## Interpretation",
        "",
        "The score is a cohort-relative ranking of observed outdoor-to-indoor coupling, not a physical air-exchange rate and not proof of a building defect. It combines best 0-12 hour lagged movement correlation (45%), response gain (20%), direction agreement (15%), split-half stable correlation (10%), and response speed (10%). Metrics use exact consecutive UTC hours and separate 1%/99% winsorization. No interpolation, TEMPO, NAQFC, weather, model predictions, or sensor substitution is used.",
        "",
        "High, moderate, and low are equal-sized thirds of the analyzed post-exclusion cohort. The percentile and exact input hashes make the cohort-relative boundary auditable. Confidence describes evidence volume and split-half availability; it does not alter the tier.",
        "",
        "## Training curriculum",
        "",
        "Use exact pair IDs, never indoor sensor ID alone. Start with `--responsiveness-tiers high`, expand with `--responsiveness-tiers high moderate`, and finish with `--responsiveness-tiers high moderate low`. Unclassified pairs are retained by the normal no-filter training path but are not silently admitted to a filtered curriculum stage.",
        "",
        "The high-correlation/high-responsiveness grouping remains useful even when response gain is the main selection criterion. Training first—or, as a controlled experiment, exclusively—on the `high` tier gives the encoder a cleaner set of repeated outdoor-to-indoor movement patterns from which to learn pattern recognition. Later expansion can test whether that representation transfers to weaker, slower, or more building-specific responses. High-only training should be compared with the staged curriculum because it deliberately reduces building diversity.",
        "",
        "The tier is a data-selection control only. It is not supplied to the model as a feature, preventing a fixed label from replacing the intended future use of longer input history to infer building response characteristics.",
        "",
        "## Reproducibility",
        "",
        f"Generated at `{summary['generated_at_utc']}`. Configuration and SHA-256 input identities are recorded in `summary.json`; per-pair evidence is in `pair_responsiveness.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    result.add_argument("--indoor-history", type=Path, action="append")
    result.add_argument("--outdoor-history", type=Path, action="append")
    result.add_argument(
        "--outdoor-exclusions", type=Path, default=DEFAULT_OUTDOOR_EXCLUSIONS
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--minimum-movements", type=int, default=168)
    result.add_argument("--maximum-lag-hours", type=int, default=12)
    result.add_argument("--winsor-percent", type=float, default=1.0)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if (
        args.minimum_movements < 2
        or args.maximum_lag_hours < 0
        or not 0 <= args.winsor_percent < 50
    ):
        raise SystemExit("invalid movement, lag, or winsor option")
    indoor_paths = args.indoor_history or list(DEFAULT_INDOOR)
    outdoor_paths = args.outdoor_history or list(DEFAULT_OUTDOOR)
    pairs = read_pairs(args.pairs)
    indoor_ids = {int(pair["indoor_sensor_id"]) for pair in pairs}
    outdoor_ids = {int(pair["outdoor_sensor_id"]) for pair in pairs}
    indoor = read_histories(indoor_paths, indoor_ids)
    outdoor = read_histories(outdoor_paths, outdoor_ids)
    exclusions = read_outdoor_exclusions(args.outdoor_exclusions)
    outdoor, excluded_hours = exclude_outdoor_readings(outdoor, exclusions)
    rows = [
        pair_metrics(
            pair,
            indoor[int(pair["indoor_sensor_id"])],
            outdoor[int(pair["outdoor_sensor_id"])],
            args.minimum_movements,
            args.maximum_lag_hours,
            args.winsor_percent,
        )
        for pair in pairs
    ]
    classify(rows)
    tiers = "high", "moderate", "low", "unclassified"
    counts = {
        tier: sum(row["responsiveness_tier"] == tier for row in rows)
        for tier in tiers
    }
    counts.update(pairs=len(rows), analyzed=len(rows) - counts["unclassified"])
    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in sorted({str(row["status"]) for row in rows})
    }
    confidence_counts = {
        confidence: sum(
            row.get("classification_confidence") == confidence for row in rows
        )
        for confidence in ("high", "moderate", "low")
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "paired PurpleAir pm2.5_atm only",
        "counts": counts,
        "status_counts": status_counts,
        "confidence_counts": confidence_counts,
        "methodology": {
            "minimum_movements": args.minimum_movements,
            "maximum_outdoor_lead_hours": args.maximum_lag_hours,
            "winsor_percent": args.winsor_percent,
            "score_weights": WEIGHTS,
            "tiers": "equal thirds of analyzed pairs",
            "confidence": {"high": "at least 1000 movements with split halves", "moderate": "at least 336 movements with split halves", "low": "remaining analyzed pairs"},
        },
        "excluded_outdoor_hours": excluded_hours,
        "inputs": [
            source_record(path)
            for path in [args.pairs, *indoor_paths, *outdoor_paths, args.outdoor_exclusions]
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pair_responsiveness.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "report.md", summary)
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
