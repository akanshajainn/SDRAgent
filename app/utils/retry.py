from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def retry_async(  # noqa: C901
    fn: Callable[[], Awaitable[T]],
    attempts: int = 3,
    base_delay_seconds: float = 0.5,
) -> T:
    """Retry an async callable with exponential backoff.

    Use this around network and tool calls where transient failures are common.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - intentionally retry all runtime failures
            last_exc = exc
            if attempt == attempts - 1:
                break
            await asyncio.sleep(base_delay_seconds * (2**attempt))
    if last_exc is None:
        raise RuntimeError("retry_async reached unreachable state with no captured exception")
    raise last_exc
