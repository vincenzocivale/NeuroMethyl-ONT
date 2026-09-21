"""Minimal cBioPortal API client for label/cohort audits.

Migrated (unchanged logic) from the sibling ``cns-io-genomics-data-audit``
project. Used for cross-checking TCGA/IPD-Brain/DBTA sample metadata against
GDC when auditing which Sturgeon classes a cohort can actually support
(docs/EXPERIMENT_POLICY.md). No NeuroMethyl-ONT-specific coupling.
"""

from __future__ import annotations

import time
from typing import Any

import requests

CBIOPORTAL_API = "https://www.cbioportal.org/api"


def _is_retriable(exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.ConnectionError | requests.exceptions.Timeout):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = exc.response
        return resp is not None and resp.status_code >= 500
    return False


def _parse_json_response(response: requests.Response, url: str) -> Any:
    content_type = response.headers.get("Content-Type", "")
    try:
        return response.json()
    except Exception as exc:
        preview = response.text[:1_000].replace("\n", "\\n")
        raise RuntimeError(
            "cBioPortal returned a non-JSON response.\n"
            f"  URL: {url}\n"
            f"  Status: {response.status_code}\n"
            f"  Content-Type: {content_type}\n"
            f"  Preview: {preview}"
        ) from exc


def get_cbioportal(
    endpoint: str, params: dict[str, Any] | None = None, retries: int = 3
) -> Any:
    url = f"{CBIOPORTAL_API}/{endpoint.lstrip('/')}"
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            response = requests.get(
                url, params=params, headers={"Accept": "application/json"}, timeout=60
            )
            response.raise_for_status()
            return _parse_json_response(response, url)
        except Exception as exc:
            last_error = exc
            if not _is_retriable(exc):
                break
            time.sleep(2**attempt)

    raise RuntimeError(f"cBioPortal GET failed after {retries} attempts: {url}") from last_error


def post_cbioportal(
    endpoint: str,
    payload: dict[str, Any],
    params: dict[str, Any] | None = None,
    retries: int = 3,
) -> Any:
    url = f"{CBIOPORTAL_API}/{endpoint.lstrip('/')}"
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                params=params,
                json=payload,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                timeout=120,
            )
            response.raise_for_status()
            return _parse_json_response(response, url)
        except Exception as exc:
            last_error = exc
            if not _is_retriable(exc):
                break
            time.sleep(2**attempt)

    raise RuntimeError(f"cBioPortal POST failed after {retries} attempts: {url}") from last_error
