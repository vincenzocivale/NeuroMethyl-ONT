from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "NEUROMETHYL_DATA_ROOT"


def get_data_root(required: bool = True) -> Path | None:
    """Return the external data root configured through the environment."""
    value = os.getenv(DATA_ROOT_ENV)
    if value:
        return Path(value).expanduser().resolve()
    if required:
        raise RuntimeError(
            f"{DATA_ROOT_ENV} is not set. Point it to the external CNSNanoporeData root."
        )
    return None
