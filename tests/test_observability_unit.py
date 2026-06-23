#!/usr/bin/env python3
"""
Unit tests for src/observability.py

Covers the Prometheus metrics wiring and the direct-to-Loki logging handler.
These are pure unit tests that don't require a running server or a live Loki.
"""

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.observability as obs


class TestHelpers:
    def test_is_truthy(self):
        assert obs._is_truthy("true")
        assert obs._is_truthy("1")
        assert obs._is_truthy("YES")
        assert obs._is_truthy("on")
        assert not obs._is_truthy("false")
        assert not obs._is_truthy(None)
        assert obs._is_truthy(None, default=True)

    def test_parse_labels(self):
        assert obs._parse_labels(None) == {}
        assert obs._parse_labels("") == {}
        assert obs._parse_labels("env=prod") == {"env": "prod"}
        # Whitespace trimmed, malformed entries skipped, empty values kept.
        assert obs._parse_labels("env=prod, service=cw ,bad,=x,k=") == {
            "env": "prod",
            "service": "cw",
            "k": "",
        }


class TestJsonLogFormatter:
    def test_basic_fields(self):
        record = logging.LogRecord("mylogger", logging.INFO, "f.py", 1, "hi %s", ("bob",), None)
        data = json.loads(obs.JsonLogFormatter().format(record))
        assert data["message"] == "hi bob"
        assert data["level"] == "INFO"
        assert data["logger"] == "mylogger"
        assert "timestamp" in data

    def test_includes_request_id_when_present(self):
        record = logging.LogRecord("l", logging.INFO, "f.py", 1, "m", (), None)
        record.request_id = "req-123"
        data = json.loads(obs.JsonLogFormatter().format(record))
        assert data["request_id"] == "req-123"

    def test_includes_exception(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord("l", logging.ERROR, "f.py", 1, "m", (), sys.exc_info())
        data = json.loads(obs.JsonLogFormatter().format(record))
        assert "ValueError: boom" in data["exception"]


class TestSetupLokiLogging:
    def test_returns_none_without_url(self, monkeypatch):
        monkeypatch.delenv("LOKI_URL", raising=False)
        assert obs.setup_loki_logging() is None

    def test_attaches_handler_when_url_set(self, monkeypatch):
        monkeypatch.setenv("LOKI_URL", "http://127.0.0.1:1/loki/api/v1/push")
        monkeypatch.setenv("LOKI_LABELS", "env=test")
        handler = obs.setup_loki_logging()
        try:
            assert isinstance(handler, obs.LokiHandler)
            assert handler in logging.getLogger().handlers
            assert handler.labels["env"] == "test"
            assert handler.labels["job"] == "claude-code-openai-wrapper"
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()


class TestLokiHandlerShipping:
    def test_ships_batched_payload_in_loki_format(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                received.append(json.loads(self.rfile.read(length)))
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()

        handler = obs.LokiHandler(
            url=f"http://127.0.0.1:{port}/loki/api/v1/push",
            labels={"job": "test"},
            batch_size=2,
            flush_interval=0.3,
        )
        handler.setFormatter(obs.JsonLogFormatter())
        log = logging.getLogger("loki-test")
        log.setLevel(logging.INFO)
        log.addHandler(handler)
        try:
            log.info("first")
            log.warning("second")

            deadline = time.monotonic() + 3
            while not received and time.monotonic() < deadline:
                time.sleep(0.05)

            assert received, "no payload reached the mock Loki server"
            streams = {s["stream"]["level"]: s for s in received[0]["streams"]}
            assert "info" in streams and "warning" in streams
            assert streams["info"]["stream"]["job"] == "test"

            ts, line = streams["info"]["values"][0]
            assert ts.isdigit()  # nanosecond timestamp
            parsed = json.loads(line)
            assert parsed["message"] == "first"
        finally:
            log.removeHandler(handler)
            handler.close()
            server.shutdown()

    def test_emit_never_raises_when_loki_unreachable(self):
        # Nothing is listening on this port; emit must not raise.
        handler = obs.LokiHandler(
            url="http://127.0.0.1:9/loki/api/v1/push",
            labels={"job": "test"},
            flush_interval=0.1,
        )
        handler.setFormatter(obs.JsonLogFormatter())
        record = logging.LogRecord("l", logging.INFO, "f.py", 1, "m", (), None)
        try:
            handler.emit(record)  # should be swallowed
            time.sleep(0.3)  # let the shipper thread try and fail quietly
        finally:
            handler.close()

    def test_httpx_transport_logs_are_not_shipped(self):
        # The handler POSTs via httpx; httpx's own request logs must be dropped to
        # avoid an endless shipping feedback loop.
        handler = obs.LokiHandler(
            url="http://127.0.0.1:9/loki/api/v1/push",
            labels={"job": "test"},
        )
        handler.setFormatter(obs.JsonLogFormatter())
        try:
            for name in ("httpx", "httpcore.connection", "httpx._client"):
                record = logging.LogRecord(name, logging.INFO, "f.py", 1, "HTTP Request", (), None)
                handler.emit(record)
            assert handler._queue.qsize() == 0

            # A normal app log is still queued.
            record = logging.LogRecord("src.main", logging.INFO, "f.py", 1, "hello", (), None)
            handler.emit(record)
            assert handler._queue.qsize() == 1
        finally:
            handler.close()


class TestSetupMetrics:
    def test_metrics_endpoint_exposed(self, monkeypatch):
        monkeypatch.delenv("METRICS_ENABLED", raising=False)
        app = FastAPI()

        @app.get("/ping")
        def ping():
            return {"ok": True}

        obs.setup_metrics(app)
        client = TestClient(app)
        client.get("/ping")
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "http_request" in resp.text

    def test_metrics_disabled(self, monkeypatch):
        monkeypatch.setenv("METRICS_ENABLED", "false")
        app = FastAPI()
        obs.setup_metrics(app)
        client = TestClient(app)
        assert client.get("/metrics").status_code == 404
