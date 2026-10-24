# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Object-to-bucket bookkeeping for multiplexed video tracking.

Multiplexing packs several tracked objects into one "bucket" so the tracker processes
``num_buckets`` joint slots instead of one slot per object. ``MultiplexState`` records
which object index sits in which bucket slot and moves arrays between data space
``(num_objects, ...)`` and multiplex space ``(num_buckets, multiplex_count, ...)``;
``MultiplexController`` creates the initial assignments.
"""

from __future__ import annotations

import logging
import math
from typing import NoReturn, cast, overload

import mlx.core as mx
import numpy as np
from mlx import nn
from numpy.typing import NDArray

from sam3_mlx.mlx_runtime import mlx_array
from sam3_mlx._unsupported import UPSTREAM_COMMIT
from sam3_mlx.mlx_runtime import is_mlx_array as _is_mlx_array


# Special values in a bucket assignment: an empty slot, and a slot whose object was
# removed from a bucket that still holds other objects. A removed slot is not reused by
# ``add_objects`` and still counts as occupied for the capacity check. Real object
# indices are >= 0.
_PADDING_NUM = -1
_REMOVED_NUM = -1116

logger = logging.getLogger(__name__)


class UnsupportedMultiplexRuntimeError(NotImplementedError):
    """Raised when a multiplex runtime component is missing from the MLX port or the supplied object."""


def raise_unsupported_multiplex_runtime(component: str) -> NoReturn:
    """Raise ``UnsupportedMultiplexRuntimeError`` naming the missing ``component``."""
    raise UnsupportedMultiplexRuntimeError(
        f"{component} is part of the official SAM3 multiplex Torch-only video "
        "runtime and is not implemented in this MLX slice. The MLX port "
        "currently provides pure multiplex helpers and fail-fast API shells; "
        "port the model path explicitly before using this component. "
        f"Upstream oracle commit: {UPSTREAM_COMMIT}."
    )


class MultiplexState[ObjectIdT = int]:
    """Records object-to-bucket assignments and converts data to and from multiplex space.

    ``assignments`` is a list of buckets, each a list of ``multiplex_count`` slots
    holding an object index (>= 0) or -1 for an empty slot. Object indices are
    consecutive from 0 and unique across buckets. At most ``allowed_bucket_capacity``
    slots per bucket may be occupied (the remaining slots always stay empty).
    Data-space arrays have leading axis ``total_valid_entries`` (one row per object);
    multiplex-space arrays have leading axes ``(num_buckets, multiplex_count)`` with
    zeros in empty slots.

    Empty state after removing all objects uses ``assignments is None`` (not ``[]``);
    callers treat that sentinel as need-reinit.

    ``device`` and ``dtype`` are accepted for API parity and are not used.
    """

    assignments: list[list[int]] | None
    object_ids: list[ObjectIdT] | None
    mux_indices_np: NDArray[np.int64]
    mux_valid_np: NDArray[np.bool_]
    demux_indices_np: NDArray[np.int64]
    mux_indices: mx.array
    mux_valid: mx.array
    demux_indices: mx.array

    def __init__(
        self,
        assignments: list[list[int]],
        device: object = None,
        dtype: object = mx.float32,
        allowed_bucket_capacity: int = 1,
        *,
        object_ids: list[ObjectIdT] | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.allowed_bucket_capacity = allowed_bucket_capacity
        self._initialize_assignments(assignments, object_ids=object_ids)

    def _require_assignments(self) -> list[list[int]]:
        """Return active assignments; empty-state sentinel is ``None``, not ``[]``."""
        if self.assignments is None:
            raise ValueError("MultiplexState has no remaining bucket assignments")
        return self.assignments

    def _initialize_assignments(
        self, assignments: list[list[int]], *, object_ids: list[ObjectIdT] | None = None
    ) -> None:
        self.assignments = assignments
        active = self._require_assignments()
        self.num_buckets = len(active)
        if self.num_buckets == 0:
            logger.error("No buckets found in the state")
            raise ValueError("No buckets found in the state")

        self.multiplex_count = len(active[0])
        assert all(len(bucket) == self.multiplex_count for bucket in active), (
            "all buckets must have the same multiplex_count"
        )

        self.total_valid_entries = sum(
            sum(1 for value in bucket if value >= 0) for bucket in active
        )
        self.total_non_padding_entries = sum(
            sum(1 for value in bucket if value != _PADDING_NUM) for bucket in active
        )

        self.object_ids = object_ids
        if self.object_ids is not None:
            assert len(self.object_ids) == self.total_valid_entries, (
                "object_ids should map 1:1 to the valid entries"
            )

        all_object_idxs: set[int] = set()
        for bucket in active:
            valid_entries_in_bucket = sum(
                1 for value in bucket if value != _PADDING_NUM
            )
            assert valid_entries_in_bucket <= self.allowed_bucket_capacity, (
                f"{valid_entries_in_bucket=} > {self.allowed_bucket_capacity=}"
            )
            for obj_idx in bucket:
                if obj_idx >= 0:
                    assert obj_idx < self.total_non_padding_entries, (
                        f"object ID {obj_idx} >= {self.total_non_padding_entries}"
                    )
                    assert obj_idx not in all_object_idxs, "object IDs must be unique"
                    all_object_idxs.add(obj_idx)

        self._precompute_transition_matrices(self.device, self.dtype)

    @property
    def available_slots(self) -> int:
        """Number of free slots across existing buckets, within ``allowed_bucket_capacity``."""
        return (
            self.num_buckets * self.allowed_bucket_capacity
            - self.total_non_padding_entries
        )

    def find_next_batch_of_available_indices(
        self,
        num_objects: int,
        *,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> list[int]:
        """Return the next ``num_objects`` unused object indices (consecutive, from ``total_valid_entries``).

        ``prefer_new_buckets`` is accepted for signature parity and ignored. Asserts
        that enough free slots exist unless ``allow_new_buckets`` is set.
        """
        del prefer_new_buckets
        assert num_objects > 0, f"{num_objects=} must be positive"
        if not allow_new_buckets:
            assert self.available_slots >= num_objects, (
                f"not enough available slots {self.available_slots} < {num_objects}"
            )
        return list(
            range(self.total_valid_entries, self.total_valid_entries + num_objects)
        )

    def add_objects(
        self,
        object_indices: list[int],
        *,
        object_ids: list[ObjectIdT] | None = None,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> None:
        """Place new objects (ascending ``object_indices``) into bucket slots.

        Empty slots of existing buckets are filled first, in order, unless
        ``prefer_new_buckets`` (which requires ``allow_new_buckets``). Objects that do
        not fit go into newly appended buckets when ``allow_new_buckets`` is set.
        ``object_ids`` must be given if and only if the state tracks ids, and is
        appended in the same order.

        Raises:
            ValueError: If objects do not fit and ``allow_new_buckets`` is False.
        """
        if len(object_indices) == 0:
            return

        object_indices = object_indices.copy()
        assert (object_ids is None) == (self.object_ids is None), (
            "object_ids must either be always given or always omitted"
        )
        pending_object_ids: list[ObjectIdT] | None = None
        if object_ids is not None:
            assert len(object_ids) == len(object_indices), (
                "object_ids must have the same length as object_indices"
            )
            pending_object_ids = object_ids.copy()

        num_new_objects = len(object_indices)
        assert object_indices == sorted(object_indices), "object_indices must be sorted"
        object_indices.reverse()
        if pending_object_ids is not None:
            pending_object_ids.reverse()

        if prefer_new_buckets:
            assert allow_new_buckets, "prefer_new_buckets requires allow_new_buckets"

        slots_filled = 0
        buckets_created = 0

        def _pop_next() -> int:
            idx = object_indices.pop()
            if pending_object_ids is not None and self.object_ids is not None:
                self.object_ids.append(pending_object_ids.pop())
            return idx

        assignments = self._require_assignments()
        if not prefer_new_buckets:
            for bucket in assignments:
                for slot_idx in range(self.allowed_bucket_capacity):
                    if bucket[slot_idx] == _PADDING_NUM:
                        bucket[slot_idx] = _pop_next()
                        slots_filled += 1
                        if len(object_indices) == 0:
                            break
                if len(object_indices) == 0:
                    break

        if len(object_indices) > 0 and not allow_new_buckets:
            raise ValueError(
                "Cannot place objects "
                f"{list(reversed(object_indices))} without creating new buckets"
            )

        while len(object_indices) > 0:
            new_bucket = [_PADDING_NUM] * self.multiplex_count
            for slot_idx in range(self.allowed_bucket_capacity):
                if len(object_indices) == 0:
                    break
                new_bucket[slot_idx] = _pop_next()
            assignments.append(new_bucket)
            buckets_created += 1

        original_num_entries = self.total_valid_entries
        self._initialize_assignments(assignments, object_ids=self.object_ids)
        assert self.total_valid_entries == original_num_entries + num_new_objects, (
            f"{self.total_valid_entries=} != {original_num_entries=} + "
            f"{num_new_objects=}"
        )

        logger.info(
            "Filled %s slots and created %s new buckets",
            slots_filled,
            buckets_created,
        )

    def remove_objects(
        self, object_indices: list[int], strict: bool = True
    ) -> list[int]:
        """Remove objects, drop buckets that become empty and renumber the rest consecutively.

        Remaining object indices keep their relative order and are remapped to
        ``0..n-1`` (``object_ids`` are remapped alongside). Returns the indices, in
        the pre-removal bucket numbering, of the buckets that were kept. If no bucket
        is left, ``assignments`` becomes ``None``. The slot of a removed object inside a
        surviving bucket is not freed for reuse. With ``strict`` an index that is not
        present fails an assertion; otherwise it is ignored.
        """
        object_indices = object_indices.copy()
        assignments = self._require_assignments()

        for bucket_idx, bucket in enumerate(assignments):
            for slot_idx, obj_id in enumerate(bucket):
                if obj_id in object_indices:
                    assignments[bucket_idx][slot_idx] = _REMOVED_NUM
                    object_indices.remove(obj_id)

        if strict:
            assert len(object_indices) == 0, (
                f"Failed to remove objects: {object_indices}"
            )

        buckets_to_remove: list[int] = []
        buckets_to_keep: list[int] = []
        for bucket_idx, bucket in enumerate(assignments):
            all_removed = all(
                obj_id in (_PADDING_NUM, _REMOVED_NUM) for obj_id in bucket
            )
            if all_removed:
                buckets_to_remove.append(bucket_idx)
            else:
                buckets_to_keep.append(bucket_idx)

        for bucket_idx in reversed(buckets_to_remove):
            del assignments[bucket_idx]

        if len(buckets_to_keep) == 0:
            self.assignments = None
            if self.object_ids is not None:
                self.object_ids = []
            return buckets_to_keep

        all_positive_ids = {
            obj_id for bucket in assignments for obj_id in bucket if obj_id >= 0
        }
        sorted_ids = sorted(all_positive_ids)
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}

        for bucket in assignments:
            for slot_idx, obj_id in enumerate(bucket):
                if obj_id >= 0:
                    bucket[slot_idx] = id_mapping[obj_id]

        if self.object_ids is not None:
            new_object_ids: list[ObjectIdT | None] = [None] * len(sorted_ids)
            for old_idx, new_idx in id_mapping.items():
                new_object_ids[new_idx] = self.object_ids[old_idx]
            assert all(obj_id is not None for obj_id in new_object_ids)
            self.object_ids = [cast(ObjectIdT, obj_id) for obj_id in new_object_ids]

        self._initialize_assignments(assignments, object_ids=self.object_ids)
        return buckets_to_keep

    def _precompute_transition_matrices(self, device: object, dtype: object) -> None:
        del device
        del dtype
        assignments = self._require_assignments()
        mux_indices = np.zeros(
            self.num_buckets * self.multiplex_count,
            dtype=np.int64,
        )
        mux_valid = np.zeros(
            self.num_buckets * self.multiplex_count,
            dtype=np.bool_,
        )
        demux_indices = np.zeros(self.total_valid_entries, dtype=np.int64)

        for bucket_idx in range(self.num_buckets):
            for slot_idx in range(self.multiplex_count):
                flat_bucket_idx = bucket_idx * self.multiplex_count + slot_idx
                object_idx = assignments[bucket_idx][slot_idx]
                if object_idx >= 0:
                    mux_indices[flat_bucket_idx] = object_idx
                    mux_valid[flat_bucket_idx] = True
                    demux_indices[object_idx] = flat_bucket_idx

        self.mux_indices_np = mux_indices
        self.mux_valid_np = mux_valid
        self.demux_indices_np = demux_indices
        self.mux_indices = mx.array(mux_indices, dtype=mx.int64)
        self.mux_valid = mlx_array(mux_valid)
        self.demux_indices = mx.array(demux_indices, dtype=mx.int64)

    @overload
    def mux(self, x: mx.array) -> mx.array: ...

    @overload
    def mux[DTypeT: np.generic](self, x: NDArray[DTypeT]) -> NDArray[DTypeT]: ...

    def mux(self, x: mx.array | NDArray[np.generic]) -> mx.array | NDArray[np.generic]:
        """Map data space ``(total_valid_entries, ...)`` to ``(num_buckets, multiplex_count, ...)``.

        Empty slots are filled with zeros. Accepts MLX or NumPy arrays and returns the
        same kind.
        """
        num_valid_entries = int(x.shape[0])
        assert num_valid_entries == self.total_valid_entries, (
            f"{num_valid_entries=} != {self.total_valid_entries=}"
        )
        output_shape = (self.num_buckets, self.multiplex_count, *tuple(x.shape[1:]))

        if num_valid_entries == 0:
            if _is_mlx_array(x):
                return mx.zeros(output_shape, dtype=x.dtype)
            return np.zeros(output_shape, dtype=np.asarray(x).dtype)

        if _is_mlx_array(x):
            x_mx = x
            gathered = mx.take(x_mx, self.mux_indices, axis=0)
            mask_shape = (*tuple(self.mux_valid.shape), *((1,) * (len(x_mx.shape) - 1)))
            return mx.reshape(
                mx.where(
                    mx.reshape(self.mux_valid, mask_shape),
                    gathered,
                    mx.zeros_like(gathered),
                ),
                output_shape,
            )

        x_np = np.asarray(x)
        gathered_np = x_np[self.mux_indices_np]
        return np.where(
            self.mux_valid_np.reshape(
                (*tuple(self.mux_valid_np.shape), *((1,) * (x_np.ndim - 1)))
            ),
            gathered_np,
            np.zeros_like(gathered_np),
        ).reshape(output_shape)

    @overload
    def demux(self, x: mx.array) -> mx.array: ...

    @overload
    def demux[DTypeT: np.generic](self, x: NDArray[DTypeT]) -> NDArray[DTypeT]: ...

    def demux(
        self, x: mx.array | NDArray[np.generic]
    ) -> mx.array | NDArray[np.generic]:
        """Map ``(num_buckets, multiplex_count, ...)`` back to ``(total_valid_entries, ...)``.

        Inverse of ``mux`` on occupied slots; the contents of empty slots are dropped.
        """
        num_buckets = int(x.shape[0])
        multiplex_count = int(x.shape[1])
        assert num_buckets == self.num_buckets, f"{num_buckets=} != {self.num_buckets=}"
        assert multiplex_count == self.multiplex_count, (
            f"{multiplex_count=} != {self.multiplex_count=}"
        )
        output_shape = (self.total_valid_entries, *tuple(x.shape[2:]))

        if self.total_valid_entries == 0:
            if _is_mlx_array(x):
                return mx.zeros(output_shape, dtype=x.dtype)
            return np.zeros(output_shape, dtype=np.asarray(x).dtype)

        if _is_mlx_array(x):
            x_mx = x
            flat = mx.reshape(
                x_mx, (num_buckets * multiplex_count, *tuple(x_mx.shape[2:]))
            )
            return mx.reshape(mx.take(flat, self.demux_indices, axis=0), output_shape)

        x_np = np.asarray(x)
        return x_np.reshape(num_buckets * multiplex_count, *x_np.shape[2:])[
            self.demux_indices_np
        ].reshape(output_shape)

    def get_valid_object_mask(self) -> mx.array:
        """Return the boolean ``(num_buckets, multiplex_count)`` occupancy mask."""
        return mx.reshape(self.mux_valid, (self.num_buckets, self.multiplex_count))

    def get_all_valid_object_idx(self) -> set[int]:
        """Return the set of object indices currently assigned to a bucket."""
        assignments = self._require_assignments()
        return {obj_idx for bucket in assignments for obj_idx in bucket if obj_idx >= 0}


class MultiplexController(nn.Module):
    """Builds initial bucket assignments for a given number of objects.

    Args:
        multiplex_count: Slots per bucket.
        full_shuffle: Place objects (and the padding) anywhere across all
            ``multiplex_count`` slots instead of only the first
            ``allowed_bucket_capacity`` of each bucket.
        eval_multiplex_count: Maximum occupied slots per bucket at inference; a
            negative value means ``multiplex_count``.
    """

    def __init__(
        self,
        multiplex_count: int,
        full_shuffle: bool = False,
        eval_multiplex_count: int = -1,
    ) -> None:
        super().__init__()
        self.multiplex_count = multiplex_count
        self.full_shuffle = full_shuffle
        self.eval_multiplex_count = (
            multiplex_count if eval_multiplex_count < 0 else eval_multiplex_count
        )
        assert self.multiplex_count >= 1

    @property
    def allowed_bucket_capacity(self) -> int:
        """Maximum number of occupied slots per bucket."""
        return self.eval_multiplex_count

    @overload
    def get_state(
        self,
        num_valid_entries: int,
        device: object = None,
        dtype: object = mx.float32,
        random: bool = True,
        *,
        object_ids: None = None,
    ) -> MultiplexState[int]: ...

    @overload
    def get_state[ObjectIdT](
        self,
        num_valid_entries: int,
        device: object = None,
        dtype: object = mx.float32,
        random: bool = True,
        *,
        object_ids: list[ObjectIdT],
    ) -> MultiplexState[ObjectIdT]: ...

    def get_state[ObjectIdT](
        self,
        num_valid_entries: int,
        device: object = None,
        dtype: object = mx.float32,
        random: bool = True,
        *,
        object_ids: list[ObjectIdT] | None = None,
    ) -> MultiplexState[ObjectIdT]:
        """Create a ``MultiplexState`` holding ``num_valid_entries`` objects.

        Uses ``ceil(num_valid_entries / allowed_bucket_capacity)`` buckets. With
        ``random`` the object-to-slot order is a random permutation drawn from NumPy's
        global random state (seed it for reproducibility); otherwise objects fill
        slots in order.
        """
        allowed_bucket_capacity = self.allowed_bucket_capacity
        true_bucket_capacity = self.multiplex_count
        num_buckets = math.ceil(num_valid_entries / allowed_bucket_capacity)

        if self.full_shuffle:
            ids = np.concatenate(
                [
                    np.arange(num_valid_entries, dtype=np.int64),
                    np.full(
                        num_buckets * true_bucket_capacity - num_valid_entries,
                        _PADDING_NUM,
                        dtype=np.int64,
                    ),
                ],
                axis=0,
            )
            if random:
                ids = ids[np.random.permutation(ids.shape[0])]
            assignments = [
                cast(
                    list[int],
                    ids[
                        bucket_idx * true_bucket_capacity : (bucket_idx + 1)
                        * true_bucket_capacity
                    ].tolist(),
                )
                for bucket_idx in range(num_buckets)
            ]
        else:
            if random:
                ids = np.random.permutation(num_valid_entries).astype(np.int64)
            else:
                ids = np.arange(num_valid_entries, dtype=np.int64)

            total_elements = num_buckets * allowed_bucket_capacity
            if ids.shape[0] < total_elements:
                ids = np.concatenate(
                    [
                        ids,
                        np.full(
                            total_elements - ids.shape[0],
                            _PADDING_NUM,
                            dtype=np.int64,
                        ),
                    ]
                )

            assignments: list[list[int]] = []
            for bucket_idx in range(num_buckets):
                bucket = cast(
                    list[int],
                    ids[
                        bucket_idx * allowed_bucket_capacity : (bucket_idx + 1)
                        * allowed_bucket_capacity
                    ].tolist(),
                )
                bucket += [_PADDING_NUM] * (
                    true_bucket_capacity - allowed_bucket_capacity
                )
                assignments.append(bucket)

        return MultiplexState(
            assignments,
            device=device,
            dtype=dtype,
            allowed_bucket_capacity=allowed_bucket_capacity,
            object_ids=object_ids,
        )
