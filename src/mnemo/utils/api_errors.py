"""Human-readable API error formatting.

Converts raw OpenAI SDK / httpx exceptions into actionable messages
that help users diagnose and fix configuration issues.
"""

from __future__ import annotations


def format_api_error(exc: Exception, *, context: str = "API") -> str:
    """Convert an API exception into a user-friendly error message.

    Parameters
    ----------
    exc : Exception
        The caught exception (typically from the ``openai`` SDK or
        an ``httpx`` transport error).
    context : str
        Human label for the failing component, e.g. ``"LLM"``
        or ``"Embedding"``.

    Returns
    -------
    str
        A one-line diagnostic message suitable for CLI display.
    """
    import openai

    # ── OpenAI / httpx API errors ──────────────────────────────────────
    if isinstance(exc, openai.APIError):
        return _format_openai_error(exc, context)

    # ── Connection / network errors ────────────────────────────────────
    msg = str(exc).lower()
    if "connection" in msg or "connect" in msg:
        return (
            f"{context}: cannot reach server — "
            f"check base_url and network connectivity"
        )
    if "timeout" in msg or "timed out" in msg:
        return (
            f"{context}: request timed out — "
            f"server not responding, check base_url"
        )
    if "name or service not known" in msg or "nodename nor servname" in msg:
        return (
            f"{context}: DNS lookup failed — "
            f"check base_url (hostname not found)"
        )

    # ── Fallback: truncate raw message ─────────────────────────────────
    short = str(exc).split("\n")[0][:150]
    return f"{context}: {short}"


def _format_openai_error(exc, context: str) -> str:
    """Format an ``openai.APIError`` subclass."""
    import openai

    status_code = getattr(exc, 'status_code', None) or 0
    message = _extract_message(exc)
    msg_lower = message.lower()

    # -- Authentication (401) -------------------------------------------
    if status_code == 401 or isinstance(exc, openai.AuthenticationError):
        if "api key" in msg_lower or "apikey" in msg_lower:
            return (
                f"{context}: invalid API key — "
                f"check the key in .env or config.toml"
            )
        return (
            f"{context}: authentication failed (401) — "
            f"check your API key"
        )

    # -- Permission (403) ------------------------------------------------
    if status_code == 403 or isinstance(exc, openai.PermissionDeniedError):
        return (
            f"{context}: access denied (403) — "
            f"check account permissions or billing status"
        )

    # -- Not found (404) -------------------------------------------------
    if status_code == 404:
        return (
            f"{context}: endpoint not found (404) — "
            f"check base_url, the path may be wrong"
        )

    # -- Rate limit (429) ------------------------------------------------
    if status_code == 429:
        return (
            f"{context}: rate limited (429) — "
            f"too many requests, wait and retry"
        )

    # -- Bad request (400) -----------------------------------------------
    if status_code == 400:
        hint = _diagnose_400(message)
        return f"{context}: bad request (400) — {hint}"

    # -- Server error (5xx) ----------------------------------------------
    if 500 <= status_code < 600:
        return (
            f"{context}: server error ({status_code}) — "
            f"the API service is having issues, retry later"
        )

    # -- Other status codes ----------------------------------------------
    return (
        f"{context}: API error ({status_code}) — "
        f"{message[:100]}"
    )


def _extract_message(exc) -> str:
    """Extract the human-readable error message from an APIError."""
    # Try the 'message' field
    msg = getattr(exc, 'message', '') or ''
    if msg:
        return str(msg)

    # Try the response body
    body = getattr(exc, 'body', None)
    if body and isinstance(body, dict):
        error = body.get('error', {})
        if isinstance(error, dict):
            return error.get('message', str(body)[:200])
        return str(body)[:200]

    # Fallback to str
    return str(exc)


def _diagnose_400(message: str) -> str:
    """Diagnose a 400 Bad Request and return a hint."""
    msg_lower = message.lower()

    # Model-related errors
    if "model" in msg_lower:
        if "not found" in msg_lower or "does not exist" in msg_lower:
            return (
                "model not found — check the model name in config.toml "
                "(is it available for this provider?)"
            )
        if "not supported" in msg_lower or "deprecated" in msg_lower:
            return (
                "model not supported — choose a different model in config.toml"
            )
        return (
            "model configuration issue — "
            "verify the model name for your provider"
        )

    # Dimension errors (embedding specific)
    if "dimension" in msg_lower:
        return (
            "unsupported dimension — "
            "check the dimension setting in config.toml"
        )

    # Generic 400
    short = message.split("\n")[0][:120]
    return f"{short} (check your config)"
