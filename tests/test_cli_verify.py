from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from embodied_sync.cli.main import main


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_verify_cli_calls_http_api_and_writes_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _Response:
        captured["document"] = json.loads(request.data)
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return _Response(
            {
                "schema_version": 1,
                "result": {
                    "verifier_id": "test-deep-api",
                    "proposed_offset_ns": 125_000_000,
                    "confidence": 0.72,
                    "details": {"model": "fake"},
                },
            }
        )

    monkeypatch.setattr("embodied_sync.inspect.verification.urlopen", fake_urlopen)
    monkeypatch.setenv("VERIFY_TEST_TOKEN", "secret")
    out = tmp_path / "review.json"
    result = main(
        [
            "verify",
            "file:///video.mp4",
            "file:///audio.wav",
            "--offset-ms",
            "100",
            "--search-radius-ms",
            "50",
            "--tolerance-ms",
            "20",
            "--api-url",
            "http://127.0.0.1:9876",
            "--token-env",
            "VERIFY_TEST_TOKEN",
            "--metadata",
            "scene=demo",
            "--out",
            str(out),
        ]
    )

    assert result == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["review"]["needs_inspection"] is True
    assert document["review"]["classical_offset_ns"] == 100_000_000
    assert document["result"]["proposed_offset_ns"] == 125_000_000
    assert captured["authorization"] == "Bearer secret"
    request = captured["document"]["request"]
    assert isinstance(request, dict)
    assert request["metadata"] == {"scene": "demo"}
    assert "wrote verification review" in capsys.readouterr().out


def test_verify_cli_requires_endpoint(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", "video", "audio", "--offset-ms", "0"]) == 2
    assert "EMBODIED_SYNC_VERIFY_URL" in capsys.readouterr().err
