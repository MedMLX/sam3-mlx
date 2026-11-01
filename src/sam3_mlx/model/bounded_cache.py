"""Small shared helpers for bounded in-process caches."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class BoundedLRUCache(Generic[K, V]):
    """Thread-safe least-recently-used map with a fixed entry bound.

    Model modules are shared across predictor sessions, so the position, RoPE,
    decoder-coordinate and compiled-function caches lock each operation. Callers
    compute values outside the lock; a lookup followed by an insertion is not
    atomic, so concurrent misses may compute and insert the same key twice.
    """

    def __init__(self, maxsize: object = 8) -> None:
        """Create an empty cache holding at most ``maxsize`` entries.

        Raises:
            ValueError: If ``maxsize`` is not a positive ``int`` (``bool`` is rejected).
        """
        if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
            raise ValueError("maxsize must be a positive integer")
        self.maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()
        self._lock = Lock()

    def get(self, key: K, default: V | None = None) -> V | None:
        """Return the value for ``key`` (marking it most recently used) or ``default``."""
        with self._lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            return self._data[key]

    def __setitem__(self, key: K, value: V) -> None:
        """Insert or replace ``key`` as the most recent entry, evicting the least recent beyond ``maxsize``."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """Remove every entry."""
        with self._lock:
            self._data.clear()
