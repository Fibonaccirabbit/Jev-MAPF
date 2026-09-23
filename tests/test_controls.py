import json
import threading
import time

import httpx
from fastapi.testclient import TestClient

from jev_mapf.policies import Policy
from functools import partial
from jev_mapf.runtime import Episode as RuntimeEpisode
Episode = partial(RuntimeEpisode, mode="centralized", obs_radius=5)
from jev_mapf.server import create_app


def test_pause_before_worker_step_prevents_call():
    ep = Episode(provider="random")
    try:
        ep.pause()
        ep.step(respect_pause=True)
        assert ep.step_count == 0
        assert ep.records == []
    finally:
        ep.close()


def test_truncated_model_output_is_not_executed():
    def respond(request):
        return httpx.Response(200, json={"id":"truncated-call", "usage":{"total_tokens":512},
            "choices":[{"finish_reason":"length", "message":{"content":'{"choice":"j22"}'}}]})
    ep = Episode(provider="chat", policy=Policy("chat", transport=httpx.MockTransport(respond)))
    try:
        result = ep.step()
        assert result["status"] == "error"
        assert result["metrics"]["steps"] == 0
        assert result["metrics"]["total_tokens"] == 512
        assert "token" in result["error"]
    finally:
        ep.close()


def test_server_is_responsive_and_reset_locked_during_model_call(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def respond(request):
        entered.set()
        assert release.wait(5)
        return httpx.Response(200, json={"choices":[{"message":{"content":'{"choice":"a2"}'}}]})
    original = Policy
    monkeypatch.setattr("jev_mapf.decentralized.Policy", lambda provider, connection, seed:
        original(provider, connection, seed, transport=httpx.MockTransport(respond)))
    with TestClient(create_app(tmp_path)) as client:
        try:
            assert client.post("/api/reset", json={"provider":"chat"}).status_code == 200
            assert client.post("/api/control/run").status_code == 200
            assert entered.wait(3)
            assert client.get("/api/state").json()["episode"]["busy"] is True
            assert client.post("/api/reset", json={}).status_code == 409
            assert client.post("/api/control/pause").status_code == 200
            release.set()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                snapshot = client.get("/api/state").json()
                if not snapshot["episode"]["busy"]:
                    break
                time.sleep(.01)
            assert snapshot["episode"]["status"] == "paused"
            assert snapshot["episode"]["metrics"]["steps"] == 0
            assert client.get("/api/export").json()["records"][0]["discarded"]
        finally:
            release.set()


def test_deepseek_sends_explicit_thinking_and_budget():
    def respond(request):
        payload = json.loads(request.content)
        assert payload["thinking"] == {"type":"enabled"}
        assert payload["reasoning_effort"] == "low"
        assert payload["max_tokens"] == 8192
        return httpx.Response(200, json={"choices":[{"message":{"content":'{"choice":"j22"}'}}]})
    ep = Episode(provider="deepseek", policy=Policy("deepseek", transport=httpx.MockTransport(respond)))
    try:
        assert ep.step()["metrics"]["steps"] == 1
    finally:
        ep.close()
