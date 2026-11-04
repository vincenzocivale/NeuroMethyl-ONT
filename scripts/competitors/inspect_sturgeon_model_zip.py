#!/usr/bin/env python3
"""Preview the contents of a Sturgeon model zip (e.g. ``general.zip``).

Migrated from the sibling ``cns-io-genomics-data-audit`` project. Prints
every archive member, previews text-like files, and lists JSON top-level
keys -- a quick sanity check before running
``extract_sturgeon_classes.py`` on the same zip.

Usage:
    python scripts/competitors/inspect_sturgeon_model_zip.py \\
        --model-zip "$NEUROMETHYL_DATA_ROOT/benchmarks/sturgeon/general.zip"
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".csv", ".tsv", ".json", ".yaml", ".yml"}


def preview_text_file(zip_file: zipfile.ZipFile, member: str, max_chars: int = 2_000) -> str:
    with zip_file.open(member) as f:
        raw = f.read(max_chars)
    try:
        return raw.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return "<could not decode as UTF-8>"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-zip", type=Path, required=True)
    args = parser.parse_args()

    if not args.model_zip.exists():
        raise FileNotFoundError(
            f"Missing model zip: {args.model_zip}. Download Sturgeon's general.zip first."
        )

    with zipfile.ZipFile(args.model_zip) as zf:
        members = zf.namelist()

        print("\nArchive members:\n")
        for member in members:
            info = zf.getinfo(member)
            print(f"- {member} ({info.file_size:,} bytes)")

        print("\nText-like file previews:\n")
        for member in members:
            if Path(member).suffix.lower() in TEXT_SUFFIXES:
                print("=" * 80)
                print(member)
                print("=" * 80)
                print(preview_text_file(zf, member))
                print()

        print("\nJSON keys:\n")
        for member in members:
            if Path(member).suffix.lower() == ".json":
                try:
                    with zf.open(member) as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        print(f"{member}: {list(data.keys())}")
                    else:
                        print(f"{member}: {type(data).__name__}")
                except Exception as exc:
                    print(f"{member}: failed to parse JSON: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
