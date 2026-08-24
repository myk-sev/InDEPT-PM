"""Create a verified snapshot of the repository's core data artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKUP_DIRECTORIES = ("data/exclusions", "data/legacy", "inputs")
BACKUP_GLOBS = (
    "data/purple air/*.csv",
    "data/purple air/k12_1km_outdoor_review/*.csv",
    "model_aware_training_balance/results/current/*",
    "purpleair_pair_exclusions/results/*.csv",
    "purpleair_pair_exclusions/results/*.json",
    "purpleair_pair_exclusions/outdoor_quality_results/*.csv",
    "purpleair_pair_exclusions/outdoor_quality_results/*.json",
)


def discover_files(root: Path = ROOT) -> list[Path]:
    """Return the existing core files in stable repository-relative order."""
    files = {
        path
        for directory in BACKUP_DIRECTORIES
        for path in (root / directory).rglob("*")
        if path.is_file()
    }
    files.update(path for pattern in BACKUP_GLOBS for path in root.glob(pattern) if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().lower())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_backup(root: Path = ROOT) -> Path:
    """Replace the single verified snapshot in ``backups``."""
    snapshot = root / "backups"
    staging = root / ".backups.partial"
    previous = root / ".backups.previous"
    if staging.exists() or previous.exists():
        raise FileExistsError("Remove .backups.partial or .backups.previous before retrying")
    staging.mkdir()

    records = []
    for source in discover_files(root):
        relative = source.relative_to(root)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256(source)
        shutil.copy2(source, target)
        if sha256(target) != source_hash:
            raise OSError(f"Backup verification failed: {relative}")
        records.append({"path": relative.as_posix(), "bytes": source.stat().st_size, "sha256": source_hash})

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if snapshot.exists():
        snapshot.rename(previous)
    try:
        staging.rename(snapshot)
    except OSError:
        if previous.exists():
            previous.rename(snapshot)
        raise
    if previous.exists():
        shutil.rmtree(previous)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List files without copying them")
    args = parser.parse_args()
    files = discover_files()

    if args.dry_run:
        for path in files:
            print(path.relative_to(ROOT))
        print(f"{len(files)} files, {sum(path.stat().st_size for path in files):,} bytes")
        return

    snapshot = create_backup(ROOT)
    print(f"Backed up {len(files)} files to {snapshot}")


if __name__ == "__main__":
    main()
