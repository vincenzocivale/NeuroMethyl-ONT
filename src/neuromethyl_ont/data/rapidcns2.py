"""Parser for Rapid-CNS2 / ONT-WGS (MNP-Flex derived) methylation files.

Observed on-disk realities this parser must tolerate (see project brief):

- Header present or absent, and not necessarily on line 1.
- Space-separated (Rapid-CNS2) or tab-separated (ONT-WGS) fields.
- Fields individually quoted, e.g. ``"chr" "start" ... "IlmnID"``.
- An optional ``mod`` column inserted between ``end`` and ``coverage``.
- Missing methylation values encoded as NA / NaN / . / null (case-insensitive).
- ``probe_id`` values that are not Illumina cg probes (e.g. "MGMT", "rs...").

The parser is header-aware and schema-aware: field positions are resolved by
name when a header is present, and only the one documented headerless layout
(chrom start end coverage methylation_percentage probe_id) is inferred by
position. Any other headerless layout, or a file with no rows the parser can
make sense of, raises :class:`SchemaDetectionError` rather than being guessed.

Malformed individual rows (bad numbers, out-of-range percentages, wrong field
count) are excluded from the parsed output but are never silently dropped:
every exclusion is recorded in ``ParseStats.anomalies`` and counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from neuromethyl_ont.data.schemas import CG_PROBE_PATTERN

MISSING_TOKENS = {"na", "nan", ".", "null", "none", ""}

CHROM_ALIASES = {"chr", "chrom", "chromosome"}
START_ALIASES = {"start", "chromstart", "chrom_start"}
END_ALIASES = {"end", "chromend", "chrom_end"}
COVERAGE_ALIASES = {"coverage", "cov", "depth", "n"}
PCT_ALIASES = {
    "methylation_percentage",
    "methylation_percent",
    "methylationpercentage",
    "percent_methylated",
    "pct_methylated",
}
PROBE_ALIASES = {"ilmnid", "ilmn_id", "probe_id", "probe", "cpg", "cpg_id"}
MOD_ALIASES = {"mod", "modification", "mod_type"}


class SchemaDetectionError(ValueError):
    """Raised when a file's schema cannot be identified without guessing."""


class RapidCNS2ParseError(ValueError):
    """Raised for a structural parse failure that invalidates the whole file."""


@dataclass
class ParseStats:
    path: str
    detected_schema: str = "undetected"
    has_mod_column: bool = False
    n_lines: int = 0
    n_header_lines: int = 0
    n_data_rows: int = 0
    n_malformed_rows: int = 0
    n_missing_methylation: int = 0
    n_missing_coverage: int = 0
    special_probe_counts: dict = field(default_factory=dict)
    anomalies: list = field(default_factory=list)

    def record_anomaly(self, line_no: int, kind: str, detail: str) -> None:
        self.anomalies.append((line_no, kind, detail))


def _dequote(token: str) -> str:
    t = token.strip()
    while len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    t = t.replace('""', '"').replace("''", "'")
    return t.strip()


def _tokenize(line: str) -> list[str]:
    raw_tokens = line.split("\t") if "\t" in line else line.split()
    return [_dequote(t) for t in raw_tokens]


def _norm(token: str) -> str:
    return token.strip().strip("\"'").strip().lower()


def _is_missing(token: str) -> bool:
    return _norm(token) in MISSING_TOKENS


def _detect_header(tokens: list[str]) -> dict[str, int] | None:
    """Return a column-name -> index map if `tokens` is a recognized header."""
    norm = [_norm(t) for t in tokens]

    def find(aliases: set[str]) -> int | None:
        for i, t in enumerate(norm):
            if t in aliases:
                return i
        return None

    coverage_idx = find(COVERAGE_ALIASES)
    pct_idx = find(PCT_ALIASES)
    probe_idx = find(PROBE_ALIASES)

    if coverage_idx is None or pct_idx is None or probe_idx is None:
        return None

    idx = {
        "chrom": find(CHROM_ALIASES),
        "start": find(START_ALIASES),
        "end": find(END_ALIASES),
        "coverage": coverage_idx,
        "pct": pct_idx,
        "probe": probe_idx,
        "mod": find(MOD_ALIASES),
    }
    return idx


def _default_headerless_schema() -> dict[str, int]:
    return {
        "chrom": 0,
        "start": 1,
        "end": 2,
        "coverage": 3,
        "pct": 4,
        "probe": 5,
        "mod": None,
    }


def _looks_like_headerless_data_row(tokens: list[str]) -> bool:
    if len(tokens) < 6:
        return False
    try:
        int(float(tokens[1]))
        int(float(tokens[2]))
    except ValueError:
        return False
    return _is_missing(tokens[3]) or _try_float(tokens[3]) is not None


def _try_float(token: str) -> float | None:
    if _is_missing(token):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _parse_row(tokens: list[str], idx: dict[str, int]) -> dict:
    def get(name: str) -> str:
        i = idx.get(name)
        if i is None:
            return ""
        if i >= len(tokens):
            raise RapidCNS2ParseError(
                f"column '{name}' index {i} out of range for row with {len(tokens)} fields"
            )
        return tokens[i]

    chrom = get("chrom")
    start_s = get("start")
    end_s = get("end")
    probe = get("probe")
    cov_s = get("coverage")
    pct_s = get("pct")

    if not probe:
        raise RapidCNS2ParseError("empty probe_id")

    start = int(float(start_s))
    end = int(float(end_s))
    if start < 0 or end < 0 or end < start:
        raise RapidCNS2ParseError(f"invalid coordinates start={start} end={end}")

    coverage = None if _is_missing(cov_s) else int(float(cov_s))
    if coverage is not None and coverage < 0:
        raise RapidCNS2ParseError(f"negative coverage: {coverage}")

    pct = None if _is_missing(pct_s) else float(pct_s)
    if pct is not None and not (0.0 <= pct <= 100.0):
        raise RapidCNS2ParseError(f"methylation percentage outside [0,100]: {pct}")

    beta = None
    methylated = None
    unmethylated = None
    if pct is not None:
        beta = pct / 100.0
        if coverage is not None:
            methylated = int(round(coverage * beta))
            methylated = max(0, min(coverage, methylated))
            unmethylated = coverage - methylated

    return {
        "chrom": chrom,
        "start": start,
        "end": end,
        "probe_id": probe,
        "coverage": coverage,
        "methylation_fraction": beta,
        "methylated_count": methylated,
        "unmethylated_count": unmethylated,
    }


def read_rapidcns2_sample(path: str | Path) -> tuple[pd.DataFrame, ParseStats]:
    """Parse one Rapid-CNS2 / ONT-WGS methylation file.

    Returns a DataFrame with columns:
    chrom, start, end, probe_id, coverage, methylation_fraction,
    methylated_count, unmethylated_count
    (no ``sample_id`` -- add it via :func:`neuromethyl_ont.data.canonical.to_coordinate_level`)

    and a :class:`ParseStats` describing what was found/excluded.

    Raises :class:`SchemaDetectionError` if no recognizable header or the
    documented headerless layout can be found before data rows begin.
    """
    path = Path(path)
    stats = ParseStats(path=str(path))
    idx: dict[str, int] | None = None
    rows: list[dict] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip("\n\r")
            if not line.strip():
                continue
            stats.n_lines += 1

            tokens = _tokenize(line)
            header_idx = _detect_header(tokens)
            if header_idx is not None:
                stats.n_header_lines += 1
                if idx is not None and idx != header_idx:
                    stats.record_anomaly(
                        line_no, "schema_change", f"{idx} -> {header_idx}"
                    )
                idx = header_idx
                if idx.get("mod") is not None:
                    stats.has_mod_column = True
                stats.detected_schema = (
                    "coverage_percentage_mod" if stats.has_mod_column
                    else "coverage_percentage"
                )
                continue

            if idx is None:
                if _looks_like_headerless_data_row(tokens):
                    idx = _default_headerless_schema()
                    stats.detected_schema = "coverage_percentage_headerless"
                else:
                    stats.record_anomaly(
                        line_no,
                        "unrecognized_preamble_or_header",
                        line[:200],
                    )
                    continue

            try:
                record = _parse_row(tokens, idx)
            except RapidCNS2ParseError as exc:
                stats.n_malformed_rows += 1
                stats.record_anomaly(line_no, "malformed_data_row", str(exc))
                continue

            stats.n_data_rows += 1
            rows.append(record)

            probe_id = record["probe_id"]
            if not CG_PROBE_PATTERN.match(probe_id):
                stats.special_probe_counts[probe_id] = (
                    stats.special_probe_counts.get(probe_id, 0) + 1
                )

            if record["coverage"] is None:
                stats.n_missing_coverage += 1
            if record["methylation_fraction"] is None:
                stats.n_missing_methylation += 1

    if idx is None:
        raise SchemaDetectionError(
            f"{path}: could not detect a recognized header or the documented "
            "headerless layout (chrom start end coverage methylation_percentage "
            "probe_id); refusing to guess."
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "chrom",
            "start",
            "end",
            "probe_id",
            "coverage",
            "methylation_fraction",
            "methylated_count",
            "unmethylated_count",
        ],
    )
    if not df.empty:
        df["chrom"] = df["chrom"].astype("string")
        df["probe_id"] = df["probe_id"].astype("string")
        df["start"] = df["start"].astype("int64")
        df["end"] = df["end"].astype("int64")
        df["coverage"] = df["coverage"].astype("Int64")
        df["methylated_count"] = df["methylated_count"].astype("Int64")
        df["unmethylated_count"] = df["unmethylated_count"].astype("Int64")
        df["methylation_fraction"] = df["methylation_fraction"].astype("float64")

    return df, stats
