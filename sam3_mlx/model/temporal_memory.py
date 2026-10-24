"""Pure temporal-index planning shared by tracker memory paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def previous_memory_frame_index(
    *,
    frame_idx: int,
    temporal_distance: int,
    stride: int,
    track_in_reverse: bool,
    selected_indices: Sequence[int] | None = None,
) -> int | None:
    """Return the frame index of the memory slot ``temporal_distance`` steps back, or None.

    With ``selected_indices`` (ordered oldest to newest) the slot is the
    ``temporal_distance``-th from the end, or None if fewer were selected.
    Otherwise the nearest slot (distance 1) is the adjacent frame in the tracking
    direction and farther slots sit on multiples of ``stride`` (rounded away from
    ``frame_idx``). Reads and mutates no tracker state.

    Raises:
        ValueError: If ``temporal_distance`` or ``stride`` is less than 1.
    """
    if temporal_distance < 1:
        raise ValueError("temporal_distance must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    if selected_indices is not None:
        if temporal_distance > len(selected_indices):
            return None
        return selected_indices[-temporal_distance]
    if temporal_distance == 1:
        direction = 1 if track_in_reverse else -1
        return frame_idx + direction
    if track_in_reverse:
        aligned = -(-(frame_idx + 2) // stride) * stride
        return aligned + (temporal_distance - 2) * stride
    aligned = ((frame_idx - 2) // stride) * stride
    return aligned - (temporal_distance - 2) * stride


@dataclass(frozen=True)
class MemoryTrimPlan:
    """Frame indices whose stored entries can be trimmed at a given frame.

    ``far_history_frame_idx`` is None when memory selection is off.
    """

    expired_frame_idx: int
    far_history_frame_idx: int | None


def memory_trim_plan(
    *,
    frame_idx: int,
    stride: int,
    num_maskmem: int,
    use_memory_selection: bool,
    max_object_pointers: int,
) -> MemoryTrimPlan:
    """Compute which frames' entries fall out of the memory window at ``frame_idx``.

    ``expired_frame_idx`` is ``stride * num_maskmem`` frames before ``frame_idx``.
    With ``use_memory_selection``, ``far_history_frame_idx`` is a further
    ``20 * max_object_pointers`` frames back (a fixed 20-frame spacing per object
    pointer). No tracker state is read or changed.

    Raises:
        ValueError: If ``stride < 1``, ``num_maskmem < 0`` or ``max_object_pointers < 0``.
    """
    if stride < 1:
        raise ValueError("stride must be positive")
    if num_maskmem < 0:
        raise ValueError("num_maskmem must be non-negative")
    if max_object_pointers < 0:
        raise ValueError("max_object_pointers must be non-negative")
    return MemoryTrimPlan(
        expired_frame_idx=frame_idx - stride * num_maskmem,
        far_history_frame_idx=(
            frame_idx - 20 * max_object_pointers if use_memory_selection else None
        ),
    )
