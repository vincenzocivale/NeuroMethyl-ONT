#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from neuromethyl_ont.data.registry import validate_dataset_registry, validate_sample_registry


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    checks = [
        validate_dataset_registry(root / "registry" / "datasets.tsv"),
        validate_sample_registry(root / "registry" / "samples.tsv"),
    ]
    errors = [error for group in checks for error in group]
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Registry schema validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
