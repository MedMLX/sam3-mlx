"""Shared stub doubles for multiplex tracker/detector test harnesses.

These stubs expose only the multiplex attributes that ``Sam3MultiplexBase``
and ``Sam3MultiplexTracking`` read during construction. The detector stub
carries only ``is_multiplex``; each test module subclasses it with the single
field its production entry point propagates (``running_in_prod`` or
``compile_model``) so that difference stays explicit at the call site.

``DummyController`` matches the official multiplex defaults used by
``test_sam3_multiplex_base``; ``test_sam3_multiplex_tracking`` keeps a richer
module-local controller (smaller bucket capacity plus ``get_state``) because
its production entry point calls it.
"""


class DummyController:
    allowed_bucket_capacity = 4
    training = False


class StubMultiplexTracker:
    is_multiplex = True


class StubMultiplexDetector:
    is_multiplex = True


class DummyTracker(StubMultiplexTracker):
    multiplex_controller = DummyController()
