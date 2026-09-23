import json

import httpx
import pytest

from jev_mapf.policies import Policy
from functools import partial
from jev_mapf.runtime import Episode as RuntimeEpisode
Episode = partial(RuntimeEpisode, mode="centralized", obs_radius=5)


@pytest.mark.parametrize("valid", [True, False])
def test_native_jev_choice_contract(valid):
    def respond(request):
        payload = json.loads(request.content)
        assert request.url.path == "/v1/systemone"
        spec = payload["questions"]["action"]
        assert spec["type"] == "choice"
        assert "j22" in spec["criteria"]
        probabilities = {k: 1.0 if valid and k == "j22" else 0.0 for k in spec["criteria"]}
        return httpx.Response(200, json={"model":"jev-test", "answers":{"action":{
            "choice":"j22", "probabilities":probabilities}}})
    ep = Episode(provider="jev", policy=Policy("jev", transport=httpx.MockTransport(respond)))
    try:
        result = ep.step()
        assert result["metrics"]["model_calls"] == 1
        assert result["metrics"]["steps"] == int(valid)
        if valid:
            assert result["last_record"]["decision"]["probabilities"]["j22"] == 1.0
        else:
            assert result["status"] == "error"
    finally:
        ep.close()
