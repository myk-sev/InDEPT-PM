"""Verify the complete current-layout bridge checkpoint family."""

from __future__ import annotations

import argparse
from pathlib import Path

from pm25_models import load_bridge_checkpoint, validate_bridge_checkpoint

from .data import file_sha256
from .models import model_names


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_ROOT = ROOT / "inference" / "checkpoints"
DEFAULT_DATASETS = (
    (
        "all_excl_fine_t",
        ROOT / "inputs" / "all_sensors_exclusion_informed_finetuned_masked_training_data.csv",
    ),
    (
        "k12_excl_fine_t",
        ROOT / "inputs" / "k12_exclusion_informed_finetuned_masked_training_data.csv",
    ),
)


def verify_family(
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    datasets: tuple[tuple[str, Path], ...] = DEFAULT_DATASETS,
) -> list[Path]:
    verified = []
    for dataset_name, training_data in datasets:
        if not training_data.is_file():
            raise FileNotFoundError(f"bridge training data not found: {training_data}")
        digest = file_sha256(training_data)
        for model_name in model_names():
            path = checkpoint_root / f"{dataset_name}_{model_name}.pt"
            checkpoint = load_bridge_checkpoint(path)
            metadata = validate_bridge_checkpoint(checkpoint, model_name)
            if metadata.get("training_data_sha256") != digest:
                raise ValueError(f"bridge training-data hash mismatch: {path}")
            if "optimizer_state" not in checkpoint:
                raise ValueError(f"bridge optimizer state missing: {path}")
            verified.append(path)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify all 22 completed bridge checkpoints."
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    args = parser.parse_args()
    paths = verify_family(args.checkpoint_root)
    print(f"verified_bridge_checkpoints={len(paths)}")
    print(f"checkpoint_root={args.checkpoint_root.resolve()}")


if __name__ == "__main__":
    main()
