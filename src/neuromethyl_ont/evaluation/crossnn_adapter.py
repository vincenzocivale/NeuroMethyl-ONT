"""Convert this project's canonical probe-level methylation into crossNN's
own native binary calling convention -- distinct from Sturgeon's dead-zone
rule (``sturgeon_adapter.py``).

crossNN's own vendored code (``third_party/crossNN/NN_model.py::
NN_classifier.predict_from_bedMethyl``) binarizes methylation at a *single*
60% cutoff with no dead zone::

    sample.loc[sample["methylation_call"] < 60, "methylation_call"] = -1
    sample.loc[sample["methylation_call"] >= 60, "methylation_call"] = 1

i.e. every covered probe gets a call -- nothing is withheld as "ambiguous".
This project's benchmark scripts previously fed crossNN through the
Sturgeon-format bed built by ``sturgeon_adapter.probe_level_to_sturgeon_bed``
(0.7/0.3, values in between dropped): that starves crossNN of evidence in
the (0.3, 0.7) range its own pipeline would still call, in every cohort this
project compares it on (RapidCNS2, ONT_WGS, the Magibu pools, GSE289246,
GSE109379, our own samples -- not GSE209865, whose GEO-published calls are
already binary with no threshold step to reproduce).

:func:`probe_level_to_crossnn_bed` reproduces crossNN's own rule directly
from this project's canonical probe-level ``methylation_fraction`` (already
a single aggregate value per probe -- see
``canonical.aggregate_to_probe_level``) -- same caveat as
``sturgeon_adapter``: not a byte-identical replay of crossNN's own
per-position bedMethyl path, the closest faithful reproduction of its
decision rule given how this project's evidence is already aggregated.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_THRESHOLD = 0.6

CROSSNN_BED_COLUMNS = ["chrom", "chromStart", "chromEnd", "methylation_call", "probe_id"]


def probe_level_to_crossnn_bed(
    probe_df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """One sample's canonical probe-level rows -> crossNN's own binary
    calling convention.

    ``probe_df`` must have ``probe_id`` and ``methylation_fraction`` columns.
    A single threshold splits every *observed* probe (``< threshold`` ->
    unmethylated, ``>= threshold`` -> methylated, matching crossNN's own
    ``>= 60`` boundary exactly) -- no dead zone, unlike
    :func:`neuromethyl_ont.evaluation.sturgeon_adapter.probe_level_to_sturgeon_bed`.
    Only a missing fraction (``NaN``, this project's "no evidence"
    convention) is dropped, never guessed. Returns
    :data:`CROSSNN_BED_COLUMNS` -- the same 5-column shape
    ``sturgeon_adapter`` produces, so ``run_crossnn_predict.py`` reads it
    unmodified.
    """
    df = probe_df[["probe_id", "methylation_fraction"]].dropna(subset=["methylation_fraction"])
    methylated = df["methylation_fraction"] >= threshold

    out = pd.DataFrame(
        {
            "chrom": "0",
            "chromStart": 0,
            "chromEnd": 1,
            "methylation_call": methylated.astype(int),
            "probe_id": df["probe_id"],
        }
    )
    return out[CROSSNN_BED_COLUMNS].reset_index(drop=True)
