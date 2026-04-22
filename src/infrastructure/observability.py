"""
Observability layer — LangFuse v3 integration for tracing, cost, and latency.

Provides:
- ``get_langfuse()``          — singleton Langfuse client
- ``fetch_prompt()``          — pull prompts from LangFuse Prompt Management
- ``observe``                 — re-exported decorator for auto-tracing
- ``update_current_trace``    — tag traces with user_id / session_id
- ``update_current_span``     — attach I/O + metadata to the current span
- ``flush()``                 — ensure events are sent before process exit

Configuration:
    .env must contain:
        LANGFUSE_SECRET_KEY
        LANGFUSE_PUBLIC_KEY
        LANGFUSE_BASE_URL   (default: https://us.cloud.langfuse.com)

    config/param.yaml:
        observability:
          enabled: true

When ``enabled`` is false every decorator becomes a no-op passthrough,
so you can turn tracing off without touching any code.
"""

import os
from typing import Optional
from infrastructure.config import settings
from infrastructure.log import get_logger

logger = get_logger(__name__, "observability.log")
# ---------------------------------------------------------------------------
# Config flag
# ---------------------------------------------------------------------------

_ENABLED: Optional[bool] = None


def _is_enabled() -> bool:
    """Check if observability is enabled (from param.yaml)."""
    global _ENABLED
    if _ENABLED is not None:
        return _ENABLED
    try:
        _ENABLED = bool(settings.observability.get("enabled", True))
    except Exception:
        _ENABLED = True
    return _ENABLED


# ---------------------------------------------------------------------------
# Singleton LangFuse client
# ---------------------------------------------------------------------------

_langfuse_client = None
_initialised = False


def get_langfuse():
    """
    Return a singleton Langfuse client.

    Returns None if observability is disabled or keys are missing.
    """
    global _langfuse_client, _initialised
    if _initialised:
        return _langfuse_client

    _initialised = True

    if not _is_enabled():
        logger.info("Observability disabled via config — LangFuse not initialised.")
        return None

    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    if not secret_key or not public_key:
        logger.warning(
            "LangFuse keys not set (LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY). "
            "Tracing is disabled."
        )
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=base_url,
        )
        logger.info("LangFuse client initialised (host=%s)", base_url)
        return _langfuse_client
    except Exception as exc:
        logger.error("Failed to initialise LangFuse: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Prompt Management — fetch from LangFuse with local fallback
# ---------------------------------------------------------------------------


def fetch_prompt(
    name: str,
    *,
    fallback: str,
    cache_ttl_seconds: int = 300,
    **compile_vars: str,
) -> str:
    """
    Fetch a prompt template from **LangFuse Prompt Management**.

    If the prompt exists in LangFuse it is compiled with ``compile_vars``
    (using ``{{variable}}`` Mustache syntax).  Otherwise the local
    ``fallback`` string is used (with Python ``{variable}`` syntax).

    This lets you edit prompts live in the LangFuse dashboard — no
    code deploy required.

    Args:
        name:  Prompt name as registered in LangFuse (e.g. ``"nawaloka-router-system"``).
        fallback:  Local prompt string used when LangFuse is unavailable
                   or the prompt hasn't been created yet.
        cache_ttl_seconds:  Client-side cache TTL (default 5 min).
        **compile_vars:  Variables to substitute into the template.

    Returns:
        Compiled prompt string ready to send to the LLM.
    """
    client = get_langfuse()

    if client is not None:
        try:
            prompt_obj = client.get_prompt(
                name,
                type="text",
                cache_ttl_seconds=cache_ttl_seconds,
            )
            compiled = prompt_obj.compile(**compile_vars) if compile_vars else prompt_obj.compile()
            logger.debug("LangFuse prompt '%s' loaded (version=%s)", name, getattr(prompt_obj, "version", "?"))
            return compiled
        except Exception as exc:
            logger.debug(
                "LangFuse prompt '%s' not found or fetch failed: %s. Using local fallback.",
                name,
                exc,
            )

    # Fallback — compile with Python str.format()
    if compile_vars:
        return fallback.format(**compile_vars)
    return fallback


# ---------------------------------------------------------------------------
# @observe decorator (re-export from langfuse — v3 API)
# ---------------------------------------------------------------------------

try:
    from langfuse import observe as _lf_observe
    from langfuse import get_client as _get_lf_client
except ImportError:
    _lf_observe = None
    _get_lf_client = None
    logger.debug("langfuse package not installed — @observe is a no-op.")


def observe(
    *,
    name: Optional[str] = None,
    as_type: Optional[str] = None,
):
    """
    Decorator that wraps ``langfuse.observe``.

    Falls back to a no-op when:
    - langfuse is not installed
    - observability is disabled in config

    Args:
        name: Span name (defaults to the function name).
        as_type: One of ``"generation"`` | ``None`` (span).
    """
    def _noop_decorator(fn):
        return fn

    if not _is_enabled() or _lf_observe is None:
        return _noop_decorator

    # Ensure client initialization happens once so missing-key or init
    # issues are logged deterministically.
    get_langfuse()

    # Build kwargs for the real decorator
    kwargs = {}
    if name is not None:
        kwargs["name"] = name
    if as_type is not None:
        kwargs["as_type"] = as_type

    return _lf_observe(**kwargs)


# ---------------------------------------------------------------------------
# Trace & Span Update Helpers (v3 API — uses get_client())
# ---------------------------------------------------------------------------


def update_current_trace(
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    tags: Optional[list] = None,
) -> None:
    """
    Update the current LangFuse trace with user/session info.

    Safe to call even when tracing is disabled (no-op).
    """
    if _get_lf_client is None or not _is_enabled():
        return
    get_langfuse()
    try:
        client = _get_lf_client()
        kwargs = {}
        if user_id is not None:
            kwargs["user_id"] = user_id
        if session_id is not None:
            kwargs["session_id"] = session_id
        if metadata is not None:
            kwargs["metadata"] = metadata
        if tags is not None:
            kwargs["tags"] = tags
        client.update_current_trace(**kwargs)
    except Exception as exc:
        logger.debug("update_current_trace failed (non-critical): %s", exc)


def update_current_observation(
    *,
    input: Optional[str] = None,
    output: Optional[str] = None,
    metadata: Optional[dict] = None,
    usage: Optional[dict] = None,
    model: Optional[str] = None,
) -> None:
    """
    Update the current span/generation with I/O and usage data.

    In LangFuse v3:
    - Generation updates use ``update_current_generation()`` with
      ``model``, ``usage_details``, ``cost_details``.
    - Span updates use ``update_current_span()`` (no model/usage).

    This helper auto-detects which to call based on whether
    ``model`` or ``usage`` are provided.

    Safe to call even when tracing is disabled (no-op).
    """
    if _get_lf_client is None or not _is_enabled():
        return
    get_langfuse()
    try:
        client = _get_lf_client()

        # If model or usage provided → generation update
        if usage is not None or model is not None:
            gen_kwargs = {}
            if input is not None:
                gen_kwargs["input"] = input
            if output is not None:
                gen_kwargs["output"] = output
            if metadata is not None:
                gen_kwargs["metadata"] = metadata
            if model is not None:
                gen_kwargs["model"] = model
            if usage is not None:
                # v3 uses usage_details (input, output, total)
                gen_kwargs["usage_details"] = usage
            try:
                client.update_current_generation(**gen_kwargs)
                return
            except Exception:
                pass

        # Otherwise → span update
        span_kwargs = {}
        if input is not None:
            span_kwargs["input"] = input
        if output is not None:
            span_kwargs["output"] = output
        if metadata is not None:
            span_kwargs["metadata"] = metadata
        if span_kwargs:
            client.update_current_span(**span_kwargs)
    except Exception as exc:
        logger.debug("update_current_observation failed (non-critical): %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def flush() -> None:
    """Flush pending LangFuse events (call before program exit)."""
    if _get_lf_client is not None and _is_enabled():
        try:
            client = _get_lf_client()
            client.flush()
            logger.debug("LangFuse flushed.")
        except Exception as exc:
            logger.debug("LangFuse flush failed: %s", exc)
