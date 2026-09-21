"""Minimal GDC (Genomic Data Commons) API client for label/cohort audits.

Migrated (unchanged logic) from the sibling ``cns-io-genomics-data-audit``
project, which used this to pair TCGA LGG/GBM methylation files with their
case/sample metadata for Sturgeon-class coverage checks
(docs/EXPERIMENT_POLICY.md). No NeuroMethyl-ONT-specific coupling -- a thin,
retrying wrapper around the public GDC REST API.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import requests

GDC_API = "https://api.gdc.cancer.gov"


DEFAULT_FILE_FIELDS = [
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "state",
    "access",
    "data_category",
    "data_type",
    "data_format",
    "experimental_strategy",
    "platform",
    "cases.case_id",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.samples.sample_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.portions.portion_id",
    "cases.samples.portions.submitter_id",
    "cases.samples.portions.slides.slide_id",
    "cases.samples.portions.slides.submitter_id",
]


def in_filter(field: str, values: list[str]) -> dict[str, Any]:
    return {"op": "in", "content": {"field": field, "value": values}}


def and_filter(filters: list[dict[str, Any]]) -> dict[str, Any]:
    return {"op": "and", "content": filters}


def post_gdc(endpoint: str, payload: dict[str, Any], retries: int = 3) -> dict[str, Any]:
    url = f"{GDC_API}/{endpoint.lstrip('/')}"

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise TypeError(f"Expected JSON object from {url}")
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)

    raise RuntimeError(f"GDC request failed after {retries} attempts") from last_error


def query_files(
    filters: dict[str, Any],
    fields: list[str] | None = None,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    selected_fields = fields or DEFAULT_FILE_FIELDS

    payload: dict[str, Any] = {
        "filters": filters,
        "fields": ",".join(selected_fields),
        "format": "JSON",
        "size": page_size,
        "from": 0,
    }

    first = post_gdc("files", payload)
    pagination = first.get("data", {}).get("pagination", {})
    total = int(pagination.get("total", 0))

    hits: list[dict[str, Any]] = list(first.get("data", {}).get("hits", []))
    while len(hits) < total:
        payload["from"] = len(hits)
        page = post_gdc("files", payload)
        hits.extend(page.get("data", {}).get("hits", []))

    return hits


def _join_unique(values: list[str]) -> str:
    return ";".join(sorted({v for v in values if v}))


def flatten_file_hit(hit: dict[str, Any]) -> dict[str, Any]:
    cases = hit.get("cases") or []

    case_ids: list[str] = []
    case_submitter_ids: list[str] = []
    project_ids: list[str] = []
    sample_ids: list[str] = []
    sample_submitter_ids: list[str] = []
    sample_types: list[str] = []
    portion_ids: list[str] = []
    portion_submitter_ids: list[str] = []
    slide_ids: list[str] = []
    slide_submitter_ids: list[str] = []

    for case in cases:
        case_ids.append(str(case.get("case_id", "")))
        case_submitter_ids.append(str(case.get("submitter_id", "")))

        project = case.get("project") or {}
        project_ids.append(str(project.get("project_id", "")))

        for sample in case.get("samples") or []:
            sample_ids.append(str(sample.get("sample_id", "")))
            sample_submitter_ids.append(str(sample.get("submitter_id", "")))
            sample_types.append(str(sample.get("sample_type", "")))

            for portion in sample.get("portions") or []:
                portion_ids.append(str(portion.get("portion_id", "")))
                portion_submitter_ids.append(str(portion.get("submitter_id", "")))

                for slide in portion.get("slides") or []:
                    slide_ids.append(str(slide.get("slide_id", "")))
                    slide_submitter_ids.append(str(slide.get("submitter_id", "")))

    return {
        "file_id": hit.get("file_id", ""),
        "file_name": hit.get("file_name", ""),
        "file_size": hit.get("file_size", ""),
        "md5sum": hit.get("md5sum", ""),
        "state": hit.get("state", ""),
        "access": hit.get("access", ""),
        "data_category": hit.get("data_category", ""),
        "data_type": hit.get("data_type", ""),
        "data_format": hit.get("data_format", ""),
        "experimental_strategy": hit.get("experimental_strategy", ""),
        "platform": hit.get("platform", ""),
        "case_ids": _join_unique(case_ids),
        "case_submitter_ids": _join_unique(case_submitter_ids),
        "project_ids": _join_unique(project_ids),
        "sample_ids": _join_unique(sample_ids),
        "sample_submitter_ids": _join_unique(sample_submitter_ids),
        "sample_types": _join_unique(sample_types),
        "portion_ids": _join_unique(portion_ids),
        "portion_submitter_ids": _join_unique(portion_submitter_ids),
        "slide_ids": _join_unique(slide_ids),
        "slide_submitter_ids": _join_unique(slide_submitter_ids),
    }


def write_gdc_manifest(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = [
        {
            "id": row["file_id"],
            "filename": row["file_name"],
            "md5": row["md5sum"],
            "size": row["file_size"],
            "state": row["state"],
        }
        for row in rows
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "filename", "md5", "size", "state"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
