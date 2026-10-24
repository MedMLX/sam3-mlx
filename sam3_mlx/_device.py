def is_mlx_runtime_device(device: object) -> bool:
    """Return whether ``device`` selects the MLX runtime (``None`` or ``"mlx"``).

    Torch device strings such as ``"cuda"`` or ``"cpu"`` are not aliases for MLX;
    callers raise an unsupported-device error for anything else.
    """
    return device in (None, "mlx")
