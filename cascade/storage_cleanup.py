"""
Audit and prune storage-heavy artifacts around cascade experiments.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPORT_SUFFIXES = {".json", ".jsonl"}
REPORT_KEYWORDS = ("bird", "gpt", "gpt-oss", "gpt_oss", "cascade", "teacher", "student")


@dataclass(frozen=True)
class CleanupTarget:
    key: str
    description: str
    caution: str
    paths: tuple[Path, ...]
    reclaimable_bytes: int
    recommended: bool

    def to_dict(self, repo_root: Path) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "caution": self.caution,
            "recommended": self.recommended,
            "reclaimable_bytes": self.reclaimable_bytes,
            "reclaimable_human": format_bytes(self.reclaimable_bytes),
            "paths": [display_path(path, repo_root) for path in self.paths],
        }


def repo_root_from(path: Path | None = None) -> Path:
    if path is not None:
        return path.resolve()
    return Path(__file__).resolve().parents[1]


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    seen_inodes: set[tuple[int, int]] = set()

    if path.is_file():
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen_inodes:
            return 0
        seen_inodes.add(inode)
        return stat.st_size

    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            stat = child.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen_inodes:
                continue
            seen_inodes.add(inode)
            total += stat.st_size
    return total


def format_bytes(num_bytes: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _existing(paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(sorted((path for path in paths if path.exists()), key=lambda item: str(item)))


def build_cleanup_targets(repo_root: Path) -> list[CleanupTarget]:
    checkpoint_dirs = _existing(
        path
        for path in (repo_root / "results" / "cascade").glob("**/checkpoints")
        if path.is_dir()
    )
    training_run_dirs = _existing(
        path
        for path in (repo_root / "data" / "lora_adapters" / "training_runs").glob("*")
        if path.is_dir()
    )
    bird_train_zip = _existing([repo_root / "bird_data" / "train" / "train_databases.zip"])
    routerbench_cache = _existing([repo_root / "data" / "routerbench_cache"])
    macos_metadata = _existing([repo_root / "bird_data" / "__MACOSX"])

    return [
        CleanupTarget(
            key="cascade_checkpoints",
            description=(
                "Delete redundant trainer checkpoint trees under results/cascade while "
                "keeping each round's exported adapter directory."
            ),
            caution=(
                "You lose trainer state and resume-from-checkpoint ability for those runs, "
                "but experiment summaries and per-round adapters remain."
            ),
            paths=checkpoint_dirs,
            reclaimable_bytes=sum(path_size_bytes(path) for path in checkpoint_dirs),
            recommended=True,
        ),
        CleanupTarget(
            key="legacy_training_runs",
            description=(
                "Delete legacy LoRA training run directories in data/lora_adapters/training_runs."
            ),
            caution=(
                "Keep this only if you still need raw trainer checkpoints beyond the registered "
                "adapters in data/lora_adapters/text_to_sql*."
            ),
            paths=training_run_dirs,
            reclaimable_bytes=sum(path_size_bytes(path) for path in training_run_dirs),
            recommended=False,
        ),
        CleanupTarget(
            key="bird_train_zip",
            description="Delete bird_data/train/train_databases.zip after extraction.",
            caution=(
                "Only the archive is removed; the extracted train_databases/ directory stays in place."
            ),
            paths=bird_train_zip,
            reclaimable_bytes=sum(path_size_bytes(path) for path in bird_train_zip),
            recommended=True,
        ),
        CleanupTarget(
            key="routerbench_cache",
            description="Delete the local RouterBench cache under data/routerbench_cache.",
            caution="Re-download will be required if you use RouterBench again.",
            paths=routerbench_cache,
            reclaimable_bytes=sum(path_size_bytes(path) for path in routerbench_cache),
            recommended=False,
        ),
        CleanupTarget(
            key="macos_metadata",
            description="Delete stray macOS extraction metadata under bird_data/__MACOSX.",
            caution="This is junk metadata and should not affect the dataset.",
            paths=macos_metadata,
            reclaimable_bytes=sum(path_size_bytes(path) for path in macos_metadata),
            recommended=True,
        ),
    ]


def find_scattered_reports(repo_root: Path) -> list[Path]:
    candidate_roots = [
        repo_root / "results",
        repo_root / "data" / "eval_results",
        repo_root / "LoRA_SGD",
    ]

    reports: list[Path] = []
    for root in candidate_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in REPORT_SUFFIXES:
                continue
            if (repo_root / "results" / "cascade") in path.parents:
                continue
            if not any(keyword in str(path).lower() for keyword in REPORT_KEYWORDS):
                continue
            reports.append(path)

    root_eval = repo_root / "eval_results.json"
    if root_eval.exists():
        reports.append(root_eval)

    return sorted(set(reports), key=lambda item: str(item))


def write_json_report(
    repo_root: Path,
    targets: Sequence[CleanupTarget],
    scattered_reports: Sequence[Path],
    output_path: Path,
) -> None:
    output = {
        "repo_root": str(repo_root),
        "targets": [target.to_dict(repo_root) for target in targets],
        "scattered_reports": [
            {
                "path": display_path(path, repo_root),
                "size_bytes": path_size_bytes(path),
                "size_human": format_bytes(path_size_bytes(path)),
            }
            for path in scattered_reports
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def print_report(
    repo_root: Path,
    targets: Sequence[CleanupTarget],
    scattered_reports: Sequence[Path],
    show_paths: bool,
) -> None:
    print(f"Cascade storage audit for {repo_root}")
    print("")
    print("Cleanup targets:")
    for target in targets:
        if not target.paths:
            continue
        recommended = "recommended" if target.recommended else "optional"
        print(
            f"- {target.key}: {format_bytes(target.reclaimable_bytes)} across "
            f"{len(target.paths)} path(s) [{recommended}]"
        )
        print(f"  {target.description}")
        print(f"  Caution: {target.caution}")
        if show_paths:
            for path in target.paths[:25]:
                print(f"    {display_path(path, repo_root)}")
            if len(target.paths) > 25:
                remaining = len(target.paths) - 25
                print(f"    ... {remaining} more path(s)")

    report_bytes = sum(path_size_bytes(path) for path in scattered_reports)
    print("")
    print(
        "Scattered JSON reports outside results/cascade: "
        f"{format_bytes(report_bytes)} across {len(scattered_reports)} file(s)"
    )
    if show_paths:
        for path in scattered_reports:
            print(f"  {display_path(path, repo_root)}")


def expand_delete_keys(
    requested_keys: Sequence[str],
    targets: Sequence[CleanupTarget],
) -> list[CleanupTarget]:
    by_key = {target.key: target for target in targets}
    if not requested_keys:
        return []

    expanded: list[CleanupTarget] = []
    for key in requested_keys:
        if key == "recommended":
            expanded.extend(target for target in targets if target.recommended and target.paths)
            continue
        try:
            expanded.append(by_key[key])
        except KeyError as exc:
            valid = ", ".join(sorted(by_key))
            raise ValueError(f"Unknown cleanup target '{key}'. Valid keys: {valid}, recommended") from exc

    seen: set[str] = set()
    unique: list[CleanupTarget] = []
    for target in expanded:
        if target.key in seen or not target.paths:
            continue
        seen.add(target.key)
        unique.append(target)
    return unique


def delete_targets(repo_root: Path, targets: Sequence[CleanupTarget]) -> None:
    repo_root_resolved = repo_root.resolve()
    target_paths = sorted(
        (path for target in targets for path in target.paths),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in target_paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(repo_root_resolved):
            raise ValueError(f"Refusing to delete path outside repo root: {resolved}")
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and optionally prune large artifacts around cascade experiments."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root to audit. Defaults to the current repo.",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print individual paths in each cleanup category.",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        default=None,
        help="Write the audit report to a JSON file.",
    )
    parser.add_argument(
        "--delete",
        nargs="+",
        default=[],
        metavar="TARGET",
        help="Delete the given target keys. Use 'recommended' to delete all recommended categories.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion. Without this flag the script only prints the planned deletions.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = repo_root_from(args.repo_root)
    targets = build_cleanup_targets(repo_root)
    scattered_reports = find_scattered_reports(repo_root)

    print_report(repo_root, targets, scattered_reports, show_paths=args.show_paths)

    if args.write_json is not None:
        write_json_report(repo_root, targets, scattered_reports, args.write_json)
        print(f"\nWrote JSON report to {args.write_json}")

    if args.delete:
        selected = expand_delete_keys(args.delete, targets)
        total_bytes = sum(target.reclaimable_bytes for target in selected)
        keys = ", ".join(target.key for target in selected)
        print(
            f"\nSelected for deletion: {keys or 'nothing'} "
            f"({format_bytes(total_bytes)} reclaimable)"
        )
        if not args.yes:
            print("Deletion not executed. Re-run with --yes to apply.")
            return 0

        delete_targets(repo_root, selected)
        print("Deletion complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
