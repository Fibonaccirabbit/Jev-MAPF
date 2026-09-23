import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from jev_mapf.benchmark_maps import load_benchmark_maps, SOURCE_COMMIT
from jev_mapf.decentralized import DecentralizedPolicy, LocalMemory
from jev_mapf.policies import Policy
from jev_mapf.runtime import Episode
from jev_mapf.server import create_app
from jev_mapf.tasks import CASES


def observation(goal=(3, 4)):
    return {"xy": np.array([0, 0]), "target_xy": np.array(goal),
            "obstacles": np.zeros((3, 3)), "agents": np.eye(3)}


def answer(payload, choice="a0"):
    criteria = payload["questions"]["action"]["criteria"]
    return httpx.Response(200, json={"model": "jev-mock", "usage": {"input_tokens": 10, "output_tokens": 2},
        "answers": {"action": {"choice": choice,
            "probabilities": {k: float(k == choice) for k in criteria}}}})


def test_local_input_whitelist_and_private_memory():
    obs = observation()
    expected, candidates = LocalMemory().prepare(obs, 64)
    polluted = {**obs, "global_map": [[9]], "all_goals": [[777, 888]], "agent_id": 19}
    actual, _ = LocalMemory().prepare(polluted, 64)
    assert actual == expected
    assert len(candidates) == 5
    assert "global_map" not in actual and "all_goals" not in actual
    assert actual["observed_map"] == ["...", "...", "..."]
    first, second = LocalMemory(), LocalMemory()
    first.prepare(obs, 64)
    obs["xy"] = np.array([5, 5])
    future, _ = first.prepare(obs, 63)
    assert any("?" in row for row in future["observed_map"])
    assert second.cells == {} and second.history == []
    first.prepare(obs, 63)
    assert first.visits[(5, 5)] == 1


def test_menu_filters_only_local_static_walls_not_other_agents():
    obs = observation()
    obs["obstacles"][0, 1] = 1
    obs["agents"][1, 2] = 1
    _, candidates = LocalMemory().prepare(obs, 64)
    menu = {c["id"]: c for c in candidates}
    assert "a1" not in menu
    assert menu["a4"]["visibly_occupied"] is True
    assert menu["a4"]["actions"] == [4]


def test_actual_jev_requests_are_isolated_and_invariant_to_other_goal():
    requests = []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert 1 <= len(payload["questions"]["action"]["criteria"]) <= 5
        assert "agents" not in payload["state"]
        assert "fixed_prefix_actions" not in payload["state"]
        assert "goals" not in payload["state"]
        return answer(payload)
    p = DecentralizedPolicy("jev", None, 42, 2, transport=httpx.MockTransport(respond))
    try:
        observations = [observation((3, 4)), observation((999, 777))]
        evidence, menus = p.prepare(observations, 64)
        p.choose(evidence, menus, observations)
        first_request = next(x for x in requests if x["state"]["own_goal"] == [3, 4])
        p.memories = [LocalMemory(), LocalMemory()]
        observations[1]["target_xy"] = np.array([-999, -777])
        requests.clear()
        evidence, menus = p.prepare(observations, 64)
        p.choose(evidence, menus, observations)
        assert first_request == next(x for x in requests if x["state"]["own_goal"] == [3, 4])
        assert p.calls == 4
        assert p.memories[0] is not p.memories[1]
        assert p.agents[0].astar is not p.agents[1].astar
    finally:
        p.close()


def test_conflicting_independent_choices_reach_environment_unmodified(monkeypatch):
    monkeypatch.setitem(CASES, "test_conflict", {"map": "...\n...\n...",
        "agents_xy": [[1, 0], [1, 2]], "targets_xy": [[1, 2], [1, 0]]})
    def respond(request):
        payload = json.loads(request.content)
        return answer(payload, "a4" if payload["state"]["own_goal"][1] > 0 else "a3")
    p = DecentralizedPolicy("jev", None, 42, 2, transport=httpx.MockTransport(respond))
    ep = Episode("test_conflict", "jev", policy=p, obs_radius=1)
    try:
        result = ep.step()
        assert result["status"] == "paused"
        assert result["last_record"]["decision"]["actions"] == [4, 3]
        # POGEMA soft resolution gives the shared cell to agent 0.
        assert result["frames"][-1]["positions"] == [[1, 1], [1, 2]]
        assert result["metrics"]["blocked_moves"] == 1
        assert result["metrics"]["model_calls"] == 2
        assert [memory.history[-1]["blocked"] for memory in p.memories] == [False, True]
    finally:
        ep.close()


def test_failed_agent_aborts_tick_and_preserves_all_call_evidence():
    def respond(request):
        payload = json.loads(request.content)
        if payload["state"]["own_goal"][1] < 0:
            return httpx.Response(401, json={"error": "secret-never-log"})
        return answer(payload, "a2")
    p = DecentralizedPolicy("jev", None, 42, 2, transport=httpx.MockTransport(respond))
    ep = Episode(provider="jev", policy=p)
    try:
        result = ep.step()
        assert result["status"] == "error"
        assert result["metrics"]["steps"] == 0
        assert result["metrics"]["model_calls"] == 2
        assert len(result["last_record"]["calls"]) == 2
        assert "secret-never-log" not in json.dumps(result)
    finally:
        ep.close()


@pytest.mark.parametrize("failure", ["connect", 429, 529])
def test_jev_transient_retry_is_bounded_and_audited(monkeypatch, failure):
    monkeypatch.setattr("jev_mapf.policies.time.sleep", lambda _: None)
    count = 0
    def respond(request):
        nonlocal count
        count += 1
        if count == 1:
            if failure == "connect":
                raise httpx.ConnectError("secret-not-logged", request=request)
            return httpx.Response(failure)
        return answer(json.loads(request.content))
    p = Policy("jev", transport=httpx.MockTransport(respond))
    evidence, candidates = LocalMemory().prepare(observation(), 64)
    try:
        result = p.choose(evidence, candidates, [observation()])
        assert result["choice"] == "a0" and p.calls == 2
        assert p.trace[0]["retry_after_seconds"] == 1
        assert p.trace[-1]["usage"]["total_tokens"] == 12
        assert "secret-not-logged" not in json.dumps(p.trace)
    finally:
        p.close()


def test_retry_exhaustion_never_selects_fallback(monkeypatch):
    monkeypatch.setattr("jev_mapf.policies.time.sleep", lambda _: None)
    p = Policy("jev", transport=httpx.MockTransport(lambda request: httpx.Response(529)))
    evidence, candidates = LocalMemory().prepare(observation(), 64)
    try:
        with pytest.raises(RuntimeError):
            p.choose(evidence, candidates, [observation()])
        assert p.calls == 3 and len(p.trace) == 3
    finally:
        p.close()


def test_official_maps_and_default_local_contract():
    with pytest.raises(ValueError, match="limited to 4"):
        Episode("validation-mazes-seed-000", mode="centralized", num_agents=8)
    maps = load_benchmark_maps()
    assert len(maps) == 145
    assert all(m["source"]["commit"] == SOURCE_COMMIT for m in maps.values())
    for case, shape in [("validation-mazes-seed-000", (21, 21)), ("wfi_warehouse", (33, 46))]:
        ep = Episode(case, num_agents=8)
        try:
            initial = ep.snapshot()
            assert initial["config"]["observation_mode"] == "decentralized_local"
            assert len(ep.policy.agents) == 8
            grid = initial["frames"][0]["map"]
            assert (len(grid), len(grid[0])) == shape
        finally:
            ep.close()


def test_web_runs_reject_centralized_mode(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/reset", json={"mode": "centralized"}).status_code == 422
        result = client.post("/api/reset", json={"case": "validation-mazes-seed-000", "num_agents": 8})
        assert result.status_code == 200
        assert client.get("/api/state").json()["episode"]["config"]["num_agents"] == 8
