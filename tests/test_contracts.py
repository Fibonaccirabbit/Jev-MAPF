import itertools
import json
import threading

import httpx
import pytest
from fastapi.testclient import TestClient
from pogema import GridConfig, pogema_v0

from jev_mapf.candidates import destinations, enumerate_candidates, legal_transition
from jev_mapf.policies import Connection, Policy
from functools import partial
from jev_mapf.runtime import Episode as RuntimeEpisode
Episode = partial(RuntimeEpisode, mode="centralized", obs_radius=5)
from jev_mapf.server import create_app


def test_candidates_exhaustive_and_match_pogema():
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    starts_list = [[[0, 0], [0, 1]], [[1, 0], [1, 2]], [[2, 1], [2, 2]]]
    for starts in starts_list:
        options = enumerate_candidates(grid, starts)
        expected = {actions for actions in itertools.product(range(5), repeat=2)
                    if legal_transition(grid, starts, destinations(starts, actions))}
        assert {tuple(c["actions"]) for c in options} == expected
        for c in options:
            env = pogema_v0(GridConfig(map=grid, agents_xy=starts, targets_xy=[[2, 0], [2, 2]],
                                      on_target="nothing", collision_system="soft"))
            env.reset()
            env.step(c["actions"])
            assert env.get_agents_xy(ignore_borders=True) == c["destinations"]
            env.close()


def test_no_edge_swaps_but_following_into_vacated_cell_is_allowed():
    menu = {c["id"] for c in enumerate_candidates([[0, 0, 0]], [[0, 0], [0, 1]])}
    assert "j43" not in menu
    assert "j40" not in menu
    assert "j44" in menu
    assert "j00" in menu


def test_baseline_has_zero_model_calls_and_classical_success(tmp_path):
    ep = Episode()
    try:
        result = ep.run()
        assert result["status"] == "success"
        assert result["metrics"]["model_calls"] == 0
        assert result["metrics"]["SoC"] == 20
        assert result["metrics"]["makespan"] == 10
        assert result["frames"][-1]["positions"] == result["frames"][-1]["goals"]
        ep.save(tmp_path)
        assert json.loads((tmp_path / "result.json").read_text())["records"]
        assert "<svg" in (tmp_path / "animation.svg").read_text()
        assert "canvas" in (tmp_path / "replay.html").read_text()
    finally:
        ep.close()


@pytest.mark.parametrize("failure", ["invalid_choice", "bad_json", "timeout", "http_error"])
def test_model_failure_never_moves_or_falls_back(failure):
    def respond(request):
        assert request.headers["authorization"] == "Bearer test-secret"
        if failure == "timeout":
            raise httpx.ReadTimeout("test", request=request)
        if failure == "http_error":
            return httpx.Response(401, json={"error": "secret body must not be logged"})
        content = "not-json" if failure == "bad_json" else '{"choice":"invented"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    policy = Policy("chat", Connection(url="http://example.test/v1/chat/completions", model="test", key="test-secret"),
                    transport=httpx.MockTransport(respond))
    ep = Episode(provider="chat", policy=policy)
    try:
        result = ep.step()
        assert result["status"] == "error"
        assert result["metrics"]["steps"] == 0
        assert result["metrics"]["model_calls"] == 1
        assert len(result["frames"]) == 1
        assert result["last_record"]["executed"] is False
        exported = json.dumps(ep.snapshot(full=True))
        assert "test-secret" not in exported
        assert "secret body" not in exported
    finally:
        ep.close()


def test_valid_model_choice_and_exact_request_are_recorded():
    def respond(request):
        sent = json.loads(request.content)
        state = json.loads(sent["messages"][1]["content"])
        assert "j22" in state["candidates"]
        assert "shortest_path" not in state
        return httpx.Response(200, json={"id": "call-123", "model": "test-model", "usage": {"total_tokens": 123},
            "choices": [{"message": {"content": '{"choice":"j22","intent":"move down"}'}}]})
    ep = Episode(provider="chat", policy=Policy("chat", transport=httpx.MockTransport(respond)))
    try:
        result = ep.step()
        assert result["frames"][-1]["positions"] == [[1, 0], [1, 5]]
        assert result["last_record"]["decision"]["probabilities"] is None
        assert result["last_record"]["calls"][0]["response_id"] == "call-123"
        assert result["metrics"]["model_calls"] == 1
    finally:
        ep.close()


def test_pause_discards_in_flight_choice():
    entered, release = threading.Event(), threading.Event()
    def respond(request):
        entered.set()
        assert release.wait(3)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"choice":"j22"}'}}]})
    ep = Episode(provider="chat", policy=Policy("chat", transport=httpx.MockTransport(respond)))
    worker = threading.Thread(target=ep.step)
    try:
        worker.start()
        assert entered.wait(3)
        assert ep.snapshot()["busy"] is True
        ep.pause()
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        result = ep.snapshot(full=True)
        assert result["status"] == "paused"
        assert result["metrics"]["steps"] == 0
        assert result["last_record"]["discarded"]
        assert result["busy"] is False
    finally:
        release.set()
        worker.join(3)
        ep.close()


def test_qwen_conditioned_choices_and_native_probabilities():
    def respond(request):
        payload = json.loads(request.content)
        context = json.loads(payload["context"])
        labels = payload["schema"]["action"]["choices"]
        label = next(k for k, v in context["labels"].items() if v == "down")
        return httpx.Response(200, json={"is_valid_json": True, "schema_match": True,
            "parsed_json": {"action": {"value": label}}, "field_telemetry": {"action": {
                "candidate_probabilities": [{"choice": k, "probability": 1.0 if k == label else 0.0} for k in labels]}}})
    ep = Episode(provider="qwen_rlcd", policy=Policy("qwen_rlcd", transport=httpx.MockTransport(respond)))
    try:
        result = ep.step()
        assert result["metrics"]["steps"] == 1
        assert result["metrics"]["model_calls"] == 2
        assert result["last_record"]["decision"]["choice"] == "j22"
        assert len(result["last_record"]["calls"]) == 2
    finally:
        ep.close()


def test_web_secrets_origins_and_validation(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.post("/api/reset", json={"max_steps": 0}).status_code == 422
    assert client.post("/api/reset", json={"case": "missing"}).status_code == 400
    assert client.post("/api/reset", json={}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.put("/api/connections/chat", json={"url":"http://localhost:1234/v1/chat/completions", "model":"test", "key":"sentinel-secret"}).status_code == 200
    assert "sentinel-secret" not in client.get("/api/config").text
    invalid = client.put("/api/connections/chat", json={"url":"invalid", "model":"test", "key":"sentinel-secret"})
    assert invalid.status_code == 422
    assert "sentinel-secret" not in invalid.text
    assert client.post("/api/reset", json={}).status_code == 200
    assert client.get("/api/export").json()["format"] == "jev-mapf-run-v1"
