from __future__ import annotations
import re
from typing import Any, Callable, Optional, Awaitable
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
    Retrying,
)
from tenacity import AsyncRetrying


def _parse_rate_limit_delay(exc: BaseException) -> Optional[float]:
    """Parse the delay from a rate limit error message, if present."""
    msg = str(exc).lower()
    match = re.search(r"try again in (\d+\.?\d*)s", msg)
    if match:
        return float(match.group(1))
    return None


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "rate limit" in msg or "429" in msg or "too many requests" in msg:
        return True
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    if status == 429:
        return True
    try:
        body = getattr(resp, "text", "") or ""
        if status == 400 and any(
            s in body.lower() for s in ["rate limit", "too many", "exceeded"]
        ):
            return True
    except Exception:
        pass
    return False


def custom_wait_strategy(retry_state) -> float:
    exc = retry_state.outcome.exception()
    if exc:
        # First, try to get the delay from the 'Retry-After' header
        response = getattr(exc, "response", None)
        if response and hasattr(response, "headers"):
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return max(1, min(float(retry_after), 600))
                except (ValueError, TypeError):
                    pass  # Fallback to other methods if parsing fails

        # If header is not present, try parsing from the error message body
        delay = _parse_rate_limit_delay(exc)
        if delay is not None:
            return max(1, min(delay, 600))

    # Fallback to exponential backoff
    return wait_random_exponential(min=1, max=600)(retry_state)


async def run_with_backoff_async(
    func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
) -> Any:
    async for attempt in AsyncRetrying(
        wait=custom_wait_strategy,
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_rate_limit_error),
        reraise=True,
    ):
        with attempt:
            return await func(*args, **kwargs)


def run_with_backoff(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    for attempt in Retrying(
        wait=custom_wait_strategy,
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_rate_limit_error),
        reraise=True,
    ):
        with attempt:
            return func(*args, **kwargs)
