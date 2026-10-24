# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Predictor wrapper that serves a SAM 3.1 multiplex model through the session API."""

from __future__ import annotations

from typing import Protocol, cast

from sam3_mlx.model.sam3_base_predictor import Sam3BasePredictor


class _WarmUpCompilationHook(Protocol):
    def __call__(self) -> object: ...


class Sam3MultiplexVideoPredictor(Sam3BasePredictor):
    """Torch-free SAM 3.1 multiplex predictor wrapper.

    The official class adds Torch autocast and optional warm-up compilation around
    the shared request/session API. The MLX port keeps the same constructor
    surface and session behavior without entering a Torch-only autocast context.

    Args:
        model: Multiplex video model implementing the session methods of
            ``Sam3BasePredictor``.
        session_expiration_sec: Idle time in seconds after which a session expires; a
            positive number.
        default_output_prob_thresh: Default score threshold for returned objects.
        async_loading_frames: Passed to the model when loading frames.
        warm_up: Run the model's ``warm_up_compilation`` hook (if it has one) at
            construction; requires a model.

    Raises:
        ValueError: If ``session_expiration_sec`` is not a positive number, or
            ``warm_up`` is set without a model.
    """

    def __init__(
        self,
        model: object | None,
        session_expiration_sec: object = 1200,
        default_output_prob_thresh: float = 0.5,
        async_loading_frames: bool = True,
        warm_up: bool = False,
    ) -> None:
        super().__init__()
        if type(session_expiration_sec) is bool or not isinstance(
            session_expiration_sec, (int, float)
        ):
            raise ValueError(
                "session_expiration_sec must be a positive number of seconds; "
                f"got {session_expiration_sec!r}."
            )
        if session_expiration_sec <= 0:
            raise ValueError(
                "session_expiration_sec must be a positive number of seconds; "
                f"got {session_expiration_sec!r}."
            )
        self.model = model
        self.session_expiration_sec = session_expiration_sec
        self.default_output_prob_thresh = default_output_prob_thresh
        self.async_loading_frames = async_loading_frames
        self.warm_up = bool(warm_up)
        if self.warm_up:
            self._run_mlx_warm_up()

    def _run_mlx_warm_up(self) -> None:
        """Call the model's optional ``warm_up_compilation`` hook, toggling ``_warm_up_complete`` around it."""
        if self.model is None:
            raise ValueError("warm_up=True requires a model instance.")
        model = self.model
        setattr(model, "_warm_up_complete", False)
        warm_up_compilation = cast(
            _WarmUpCompilationHook | None,
            getattr(model, "warm_up_compilation", None),
        )
        if warm_up_compilation is not None:
            warm_up_compilation()
        setattr(model, "_warm_up_complete", True)
