"""Observability helpers: Prometheus metrics and direct Loki log shipping.

Both integrations are opt-in and configured entirely via environment variables so
the wrapper behaves identically to before when they are not set:

- ``METRICS_ENABLED`` (default ``true``)   Expose Prometheus metrics at ``/metrics``.
- ``LOKI_URL``                              Loki push endpoint, e.g.
                                            ``http://loki:3100/loki/api/v1/push``.
                                            When set, logs are pushed directly to Loki.
- ``LOKI_LABELS``                           Extra static stream labels as a comma
                                            separated ``key=value`` list, e.g.
                                            ``env=prod,service=claude-wrapper``.
- ``LOKI_LOG_LEVEL``                        Minimum level shipped to Loki (default ``INFO``).
"""

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


def _is_truthy(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def _parse_labels(raw: Optional[str]) -> Dict[str, str]:
    """Parse a ``key=value,key2=value2`` string into a dict, skipping bad entries."""
    labels: Dict[str, str] = {}
    if not raw:
        return labels
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            labels[key] = value
    return labels


class JsonLogFormatter(logging.Formatter):
    """Render log records as single-line JSON so Loki's ``json`` parser can index them."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Surface a request id when middleware attached one via `extra={...}`.
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# Entry tuple: (loki_timestamp_ns, formatted_line, level_label)
_LogEntry = Tuple[str, str, str]

# Loggers whose records must never be shipped to Loki. The Loki handler itself
# POSTs via httpx, and httpx/httpcore log each request; shipping those records
# would create a self-sustaining feedback loop (every push generates a new log
# line to push). These are dropped from Loki regardless of log level.
_EXCLUDED_LOGGER_PREFIXES = ("httpx", "httpcore")


class LokiHandler(logging.Handler):
    """A logging handler that batches records and pushes them to Loki over HTTP.

    Shipping happens on a dedicated daemon thread fed by an in-memory queue, so
    application request handlers never block on Loki I/O. If Loki is unreachable the
    failure is swallowed (logging must never crash the app) and records are dropped
    once the bounded queue is full.
    """

    def __init__(
        self,
        url: str,
        labels: Dict[str, str],
        batch_size: int = 100,
        flush_interval: float = 2.0,
        timeout: float = 5.0,
        queue_size: int = 10000,
    ) -> None:
        super().__init__()
        self.url = url
        self.labels = labels or {}
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue: "queue.Queue[_LogEntry]" = queue.Queue(maxsize=queue_size)
        self._client = httpx.Client(timeout=timeout)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="loki-shipper", daemon=True)
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        # Never ship transport logs from our own push client back to Loki, which
        # would loop endlessly.
        if record.name.startswith(_EXCLUDED_LOGGER_PREFIXES):
            return
        try:
            entry: _LogEntry = (
                str(int(record.created * 1_000_000_000)),
                self.format(record),
                record.levelname.lower(),
            )
            self._queue.put_nowait(entry)
        except queue.Full:
            # Drop rather than block the caller when the buffer is saturated.
            pass
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)

    def _run(self) -> None:
        batch: List[_LogEntry] = []
        last_flush = time.monotonic()
        while not self._stop.is_set():
            wait = max(0.0, self.flush_interval - (time.monotonic() - last_flush))
            try:
                batch.append(self._queue.get(timeout=wait))
            except queue.Empty:
                pass
            if batch and (
                len(batch) >= self.batch_size
                or (time.monotonic() - last_flush) >= self.flush_interval
            ):
                self._ship(batch)
                batch = []
                last_flush = time.monotonic()
        # Drain whatever is left on shutdown.
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._ship(batch)

    def _ship(self, batch: List[_LogEntry]) -> None:
        # Group lines into one stream per level so the `level` label stays useful
        # without exploding label cardinality.
        streams: Dict[str, List[List[str]]] = {}
        for ts, line, level in batch:
            streams.setdefault(level, []).append([ts, line])
        payload = {
            "streams": [
                {"stream": {**self.labels, "level": level}, "values": values}
                for level, values in streams.items()
            ]
        }
        try:
            response = self._client.post(self.url, json=payload)
            response.raise_for_status()
        except Exception:  # noqa: BLE001 - never let log shipping crash the app
            pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
        super().close()


def setup_loki_logging() -> Optional[LokiHandler]:
    """Attach a Loki push handler to the root logger when ``LOKI_URL`` is configured."""
    loki_url = os.getenv("LOKI_URL")
    if not loki_url:
        return None

    labels = {"job": os.getenv("LOKI_JOB", "claude-code-openai-wrapper")}
    labels.update(_parse_labels(os.getenv("LOKI_LABELS")))

    handler = LokiHandler(url=loki_url, labels=labels)
    handler.setFormatter(JsonLogFormatter())
    level_name = os.getenv("LOKI_LOG_LEVEL", "INFO").upper()
    handler.setLevel(getattr(logging, level_name, logging.INFO))

    # The push client emits an httpx "HTTP Request" log on every flush. Quiet it
    # to WARNING so log shipping doesn't spam the console with its own traffic.
    for noisy in _EXCLUDED_LOGGER_PREFIXES:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger().addHandler(handler)
    logger.info("Loki log shipping enabled -> %s (labels=%s)", loki_url, labels)
    return handler


def setup_metrics(app) -> None:
    """Expose Prometheus metrics at ``/metrics`` unless disabled via ``METRICS_ENABLED``."""
    if not _is_truthy(os.getenv("METRICS_ENABLED"), default=True):
        logger.info("Prometheus metrics disabled (METRICS_ENABLED is false)")
        return

    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        logger.warning(
            "prometheus-fastapi-instrumentator not installed; /metrics endpoint unavailable"
        )
        return

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
    ).instrument(
        app
    ).expose(app, endpoint="/metrics", include_in_schema=False)
    logger.info("Prometheus metrics enabled at /metrics")
