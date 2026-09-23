from fastapi.testclient import TestClient

from functools import partial
from jev_mapf.runtime import Episode as RuntimeEpisode
Episode = partial(RuntimeEpisode, mode="centralized", obs_radius=5)
from jev_mapf.server import create_app


def test_collision_resolution_cannot_rewrite_requested_actions():
    ep = Episode()
    try:
        ep.policy.astar.act = lambda observations: [1, 1]
        record = ep.step()["last_record"]
        assert record["decision"]["choice"] == "j11"
        assert record["decision"]["actions"] == [1, 1]
        assert record["after"]["positions"] == record["before"]["positions"]
        assert ep.snapshot()["metrics"]["blocked_moves"] == 2
    finally:
        ep.close()


def test_saved_run_replay_and_path_boundaries(tmp_path):
    ep = Episode()
    try:
        ep.run()
        ep.save(tmp_path / "saved-case")
    finally:
        ep.close()
    with TestClient(create_app(tmp_path)) as client:
        rows = client.get("/api/runs").json()
        assert rows[0]["name"] == "saved-case"
        saved = client.get("/api/runs/saved-case").json()
        assert saved["archive_name"] == "saved-case"
        assert saved["metrics"]["CSR"] == 1.0
        assert len(saved["frames"]) == 11
        assert "canvas" in client.get("/api/runs/saved-case/animation").text
        assert client.get("/api/runs/.hidden").status_code == 400
        assert client.get("/api/runs/missing").status_code == 404
        assert client.get("/api/state").json()["episode"] is None
