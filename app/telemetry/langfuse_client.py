from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from langfuse import Langfuse
from langfuse._client.span import LangfuseGeneration, LangfuseSpan
from langfuse.types import TraceContext

from app.utils.logging import get_logger

if os.getenv("ENV") != "production":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

logger = get_logger(__name__)

_langfuse_client: Optional[Langfuse] = None
_client_disabled: bool = False


def _get_langfuse_client() -> Optional[Langfuse]:
    """Return a singleton Langfuse client if credentials are configured."""
    global _langfuse_client, _client_disabled

    if _client_disabled:
        return None
    if _langfuse_client:
        return _langfuse_client

    # Simple, explicit env var usage for v1: read the documented names only.
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL")

    # Require both public and secret keys to enable telemetry in this v1.
    if not public_key or not secret_key:
        _client_disabled = True
        logger.info(
            "Langfuse telemetry disabled: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set to enable telemetry."
        )
        return None

    try:
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            environment=os.getenv("ENV") or "prod",
        )
    except Exception as exc:  # pragma: no cover - defensive
        _client_disabled = True
        logger.warning("Failed to initialize Langfuse telemetry client: %s", exc)
        return None

    return _langfuse_client


def _build_trace_context(span: Optional[LangfuseSpan]) -> Optional[TraceContext]:
    """Convert a span into a TraceContext usable for child generations."""
    if not span:
        return None
    return TraceContext(trace_id=span.trace_id, parent_span_id=span.id)


def start_request_trace(
    name: str,
    input_payload: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[LangfuseSpan]:
    """Start a Langfuse span that represents the overall API request."""
    client = _get_langfuse_client()
    if not client:
        return None
    try:
        return client.start_span(name=name, input=input_payload, metadata=metadata)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unable to start Langfuse request trace '%s': %s", name, exc)
        return None


def finalize_request_trace(
    span: Optional[LangfuseSpan],
    output_payload: Optional[Dict[str, Any]] = None,
    error: Optional[BaseException] = None,
) -> None:
    """Update and close the request span."""
    if not span:
        return
    level = "ERROR" if error else None
    status_message = str(error) if error else None
    try:
        span.update(
            output=output_payload,
            level=level,  # type: ignore[arg-type]
            status_message=status_message,
        )
        span.end()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to finalize Langfuse request trace: %s", exc)


def start_provider_generation(
    parent_span: Optional[LangfuseSpan],
    *,
    name: str,
    model: Optional[str],
    input_payload: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[LangfuseGeneration]:
    """Start a generation/span for a specific provider/model invocation."""
    client = _get_langfuse_client()
    trace_context = _build_trace_context(parent_span)
    if not client or not trace_context:
        return None
    try:
        return client.start_observation(
            as_type="generation",
            trace_context=trace_context,
            name=name,
            model=model,
            input=input_payload,
            metadata=metadata,
            completion_start_time=datetime.now(timezone.utc),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unable to start Langfuse generation for '%s': %s", name, exc)
        return None


def finalize_provider_generation(
    generation: Optional[LangfuseGeneration],
    *,
    output_payload: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    usage_details: Optional[Dict[str, int]] = None,
    cost_details: Optional[Dict[str, float]] = None,
    error: Optional[BaseException] = None,
) -> None:
    """Update and close a provider generation span."""
    if not generation:
        return
    level = "ERROR" if error else None
    status_message = str(error) if error else None
    try:
        generation.update(
            output=output_payload,
            metadata=metadata,
            usage_details=usage_details,
            cost_details=cost_details,
            level=level,  # type: ignore[arg-type]
            status_message=status_message,
        )
        generation.end()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to finalize Langfuse generation: %s", exc)


def flush_telemetry() -> None:
    """Force-flush buffered Langfuse events for near real-time visibility.

    Warning: This is a blocking call that waits for queues to drain.
    Use flush_telemetry_async() in async contexts to avoid blocking.
    """
    client = _get_langfuse_client()
    if not client:
        return
    try:
        client.flush()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Langfuse flush failed: %s", exc)


async def flush_telemetry_async() -> None:
    """Non-blocking async version of flush_telemetry.

    Runs the blocking flush in a thread pool to avoid blocking the event loop.
    """
    import asyncio

    await asyncio.to_thread(flush_telemetry)
