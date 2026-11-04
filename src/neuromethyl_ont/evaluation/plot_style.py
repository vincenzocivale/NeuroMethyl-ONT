"""Shared colours for the benchmark figures (one colour per model, fixed)."""

from __future__ import annotations

INK = "#1a1a1a"
MUTED = "#6b6b66"
HAIRLINE = "#e4e2dc"

# Our models sit on one blue ramp; competitors use non-blue hues so they are
# never confused with ours.
COLOR = {
    "Binary": "#3987e5",
    "Beta": "#9ec5f4",
    "Evidence": "#184f95",
    "Consensus": "#0d366b",
    "Sturgeon": "#eb6834",
    "crossNN": "#8a8a85",
    "MethyLYZR": "#5ea86d",
}
