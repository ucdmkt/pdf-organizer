"""Resilience patterns (retries, timeouts) for API interactions."""

import httpcore
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

_MAX_ATTEMPTS = 100
_MIN_SLEEP = 60


def should_retry_on_exception(exception):
    """
    Checks if the exception is one that should be retried with backoff.
    This includes 429 Resource Exhausted errors and transient network/connection issues.
    """
    # 1. Check for 429 / Resource Exhausted
    code = getattr(exception, "status_code", None)
    if code is None:
        code = getattr(exception, "code", None)
    if code is None:
        code = getattr(exception, "status", None)

    if code in [429, "429"]:
        return True

    err_str = str(exception).upper()
    if (
        "429" in err_str
        or "RESOURCE_EXHAUSTED" in err_str
        or "RESOURCE EXHAUSTED" in err_str
    ):
        return True

    # 2. Check for network/connection/timeout errors (httpx and httpcore)
    if isinstance(
        exception, (httpx.TimeoutException, httpx.ConnectError, httpcore.NetworkError)
    ):
        return True

    # 3. Check for typical network error messages in strings
    if any(
        msg in err_str
        for msg in [
            "CONNECTION TIMED OUT",
            "TIMEOUT",
            "CONNECTION ERROR",
            "NETWORK ERROR",
        ]
    ):
        return True

    return False


# Shared retry strategy for transient errors (429 and network)
def retry_with_backoff(logger):
    return retry(
        retry=retry_if_exception(should_retry_on_exception),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1.5, min=_MIN_SLEEP, max=3600),
        before_sleep=lambda retry_state: logger.warning(
            f"⚠️ Transient error met. Retrying... "
            f"Attempt {retry_state.attempt_number}. "
            f"Error: {retry_state.outcome.exception()}"
        ),
    )
