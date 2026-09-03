"""
MOST (Maynard Operation Sequence Technique) index/TMU scale.

This project does NOT perform genuine MOST analysis -- it has no distance,
weight, or other physical parameter to look up in an index table, because
segments here come from a VLM watching a video, not from measured motion
parameters. What it CAN do is express a segment's *observed* duration on
the same index/TMU scale MOST uses, for comparability. See README.md for
why that distinction matters.

TMU = index x 10. 1 TMU = 0.00001 hour = 0.036 second. The index
progression (0, 1, 3, 6, 10, 16, 24, ...) is the standard MOST scale used
across every parameter table (distance, weight, time, etc).
"""

from __future__ import annotations

TMU_SECONDS = 0.036
MOST_INDEX_VALUES = [0, 1, 3, 6, 10, 16, 24, 32, 42, 54, 70, 90, 110, 135, 165, 200, 240, 290, 350, 420]


def snap_to_index(raw_value: float, table: list[int] = MOST_INDEX_VALUES) -> int:
    if raw_value <= 0:
        return 0
    if raw_value <= table[-1]:
        return min(table, key=lambda v: abs(v - raw_value))
    v = table[-1]
    while v < raw_value:
        v = round(v * 1.2)
    return v


def index_to_tmu(index: int) -> float:
    return index * 10


def tmu_to_seconds(tmu: float) -> float:
    return tmu * TMU_SECONDS


def duration_to_index(duration_s: float) -> int:
    """Time-equivalent index for a segment with no distance parameter:
    index = TMU/10 = (duration_s / 0.036) / 10, snapped to the standard
    index scale."""
    return snap_to_index(duration_s / (TMU_SECONDS * 10))
