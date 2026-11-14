# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Retry policy: exponential backoff with full jitter, a bounded number of attempts and a total
time budget. Only errors marked `retryable` are retried, and only for calls that are safe to
repeat (the client decides which; see `ombra._transport.may_retry`). Needs neither grpc nor the
stubs.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

from ombra.errors import OmbraError

log = logging.getLogger("ombra.retry")
T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """How a call that is safe to repeat is repeated.

    `max_attempts` counts the first try (1 = never retry). Before attempt n+1 the client sleeps a
    random time in [0, min(max_delay, base_delay * 2**(n-1))] (full jitter). No new attempt starts
    when the time since the first one plus the next sleep would pass `total_budget` seconds.
    """

    max_attempts: int = 4
    base_delay: float = 0.25
    max_delay: float = 8.0
    total_budget: float = 120.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0 or self.total_budget < 0:
            raise ValueError("delays and the budget must not be negative")

    @classmethod
    def never(cls) -> RetryPolicy:
        """A policy that makes one attempt only."""
        return cls(max_attempts=1)

    def delay_bound(self, attempt: int) -> float:
        """Largest sleep after failed attempt number `attempt` (1-based)."""
        return min(self.max_delay, self.base_delay * 2 ** (attempt - 1))


class Replayable:
    """A request stream that can be sent again: wraps a function that makes a fresh iterator.

    A plain generator is used up after one attempt, so calls that stream requests pass one of
    these; the transport iterates it anew for each attempt.
    """

    def __init__(self, factory: Callable[[], Iterator[Any]]) -> None:
        self._factory = factory

    def __iter__(self) -> Iterator[Any]:
        return iter(self._factory())


def call_with_retry(
    fn: Callable[[], T],
    *,
    rpc: str,
    policy: RetryPolicy,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    """Run `fn`, repeating it while it raises a retryable OmbraError within the policy's limits.
    Logs each retry at debug level with the call name, attempt number and status code only."""
    start = clock()
    attempt = 1
    while True:
        try:
            return fn()
        except OmbraError as err:
            if not err.retryable or attempt >= policy.max_attempts:
                raise
            delay = rand() * policy.delay_bound(attempt)
            if clock() - start + delay >= policy.total_budget:
                raise
            log.debug("retry %s attempt=%d status=%s", rpc, attempt, err.grpc_status)
            sleep(delay)
            attempt += 1


def retrying(stub: Any, rpc: str, fn: Callable[[], T]) -> T:
    """Run `fn` under the retry policy of `stub`, or once if the stub has none (local mode)."""
    runner = getattr(stub, "retry_runner", None)
    return fn() if runner is None else runner(rpc, fn)
