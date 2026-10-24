import logging

import pytest

from sam3_mlx.logger import get_logger


def test_invalid_log_level_env_warns_and_keeps_default_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "NOISY")
    with pytest.warns(UserWarning, match="invalid LOG_LEVEL"):
        logger = get_logger("sam3_mlx_test_invalid_env", level=logging.INFO)
    assert logger.level == logging.INFO


def test_get_logger_updates_existing_handler_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    name = "sam3_mlx_test_handler_level"

    first = get_logger(name, level=logging.INFO)
    assert first.level == logging.INFO

    second = get_logger(name, level=logging.DEBUG)
    assert second.level == logging.DEBUG
    handlers = [
        handler
        for handler in second.handlers
        if type(handler).__name__ == "_ColoredStreamHandler"
    ]
    assert len(handlers) == 1
    assert handlers[0].level == logging.DEBUG
