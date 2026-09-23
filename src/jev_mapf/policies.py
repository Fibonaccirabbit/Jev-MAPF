"""Model transports and explicit baselines. Transport failures never select an action."""

import json
import math
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np
from pydantic import BaseModel, Field, model_validator
from pogema import BatchAStarAgent

from .candidates import NAMES, SYSTEM

PROVIDERS = ("astar", "random", "deepseek", "chat", "qwen_rlcd", "jev")


def instructions_for(evidence):
    if evidence.get("observation_mode") == "decentralized_local":
        from .decentralized import COOP_NOTE, LOCAL_SYSTEM
        if "group_plan" in evidence:
            return LOCAL_SYSTEM.replace(" Return only JSON:", COOP_NOTE + " Return only JSON:")
        return LOCAL_SYSTEM
    return SYSTEM


class Connection(BaseModel):
    url: str = "https://api.deepseek.com/chat/completions"
    model: str = "deepseek-flash"
    key: str = Field(default="", repr=False)
    thinking: bool = True
    max_tokens: int = Field(default=8192, ge=256, le=16384)

    @model_validator(mode="after")
    def validate_url(self):
        p = urlsplit(self.url)
        if (p.scheme not in {"http", "https"} or not p.hostname or p.username
                or p.password or p.query or p.fragment):
            raise ValueError("Use a plain HTTP(S) endpoint without credentials or query parameters")
        if not 1 <= len(self.model) <= 256:
            raise ValueError("model must contain 1..256 characters")
        return self

    def public(self):
        return {"url": self.url, "model": self.model, "has_key": bool(self.key),
                "thinking": self.thinking, "max_tokens": self.max_tokens}


def default_connection(provider):
    if provider == "qwen_rlcd":
        return Connection(url=os.getenv("QWEN_RLCD_URL", "http://127.0.0.1:8000/api/run-rlcd"),
                          model="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    if provider == "jev":
        return Connection(url="https://api.typesafe.ai/v1/systemone", model="jev-latest",
                          key=os.getenv("TYPESAFE_API_KEY", ""))
    if provider == "chat":
        return Connection(url=os.getenv("MAPF_API_URL", "http://127.0.0.1:8000/v1/chat/completions"),
                          model=os.getenv("MAPF_API_MODEL", "local-model"), key=os.getenv("MAPF_API_KEY", ""))
    return Connection(key=os.getenv("DEEPSEEK_API_KEY", ""),
                      model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))


def import_embodied_deepseek():
    """Read only the explicitly requested saved DeepSeek credential; never persist it here."""
    import keyring
    path = Path.home() / "Library/Application Support/EmbodiedJev/connections.json"
    metadata = json.loads(path.read_text())
    item = metadata.get("connections", {}).get("deepseek")
    prefix = "connection:deepseek:"
    if item is None:
        profiles = [(identity, value) for identity, value in metadata.get("profiles", {}).items()
                    if value.get("provider") == "deepseek"]
        if len(profiles) != 1:
            raise ValueError("A unique saved DeepSeek connection is required")
        identity, item = profiles[0]
        if not re.fullmatch(r"[a-f0-9]{12}", identity):
            raise ValueError("Invalid saved profile ID")
        prefix = f"profile:{identity}:"
    if item["url"] != "https://api.deepseek.com/chat/completions":
        raise ValueError("Saved DeepSeek endpoint is not the official endpoint")
    if not re.fullmatch(re.escape(prefix) + r"[a-f0-9]{32}", item.get("secret_ref", "")):
        raise ValueError("Saved credential reference is invalid")
    key = keyring.get_password("EmbodiedJev", item["secret_ref"])
    if not key:
        raise ValueError("Saved DeepSeek credential is unavailable")
    return Connection(url=item["url"], model=item["model"], key=key)


def validate_probabilities(values, choices):
    if not isinstance(values, dict) or set(values) != set(choices):
        raise ValueError("Provider must return a probability for every offered candidate")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for v in values.values()) or abs(sum(values.values()) - 1) > 0.02:
        raise ValueError("Invalid candidate probability distribution")
    return values


class Policy:
    def __init__(self, provider, connection=None, seed=42, transport=None):
        if provider not in PROVIDERS:
            raise ValueError("Unknown provider")
        self.provider = provider
        self.connection = connection or default_connection(provider)
        if provider == "deepseek" and self.connection.url != "https://api.deepseek.com/chat/completions":
            raise ValueError("DeepSeek credentials may only be sent to the official endpoint")
        if provider == "jev" and self.connection.url != "https://api.typesafe.ai/v1/systemone":
            raise ValueError("Jev credentials may only be sent to the official endpoint")
        self.client = httpx.Client(timeout=120, follow_redirects=False, transport=transport)
        self.astar = BatchAStarAgent()
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.trace = []

    def close(self):
        self.client.close()

    def _post(self, payload):
        # Retry transport/transient service failures, never invalid model decisions.
        attempts = max(1, min(int(os.getenv("JEV_MAX_ATTEMPTS", "3")), 10)) if self.provider == "jev" else 1
        for attempt in range(1, attempts + 1):
            try:
                return self._post_once(payload)
            except RuntimeError:
                trace = self.trace[-1]
                transient = (trace.get("error_type") in {
                    "ConnectError", "ConnectTimeout", "ReadError", "ReadTimeout",
                    "WriteError", "WriteTimeout", "RemoteProtocolError"}
                    or trace.get("status_code") in {429, 500, 502, 503, 504, 529})
                trace["attempt"] = attempt
                if not transient or attempt == attempts:
                    raise
                delay = min(2 ** (attempt - 1), 8)
                trace["retry_after_seconds"] = delay
                time.sleep(delay)

    def _post_once(self, payload):
        self.calls += 1
        trace = {"endpoint": self.connection.url, "request": payload}
        self.trace.append(trace)
        started = time.perf_counter()
        try:
            headers = {"Authorization": f"Bearer {self.connection.key}"} if self.connection.key else {}
            response = self.client.post(self.connection.url, json=payload, headers=headers)
            trace["status_code"] = response.status_code
            if response.status_code != 200:
                raise RuntimeError(f"Model HTTP {response.status_code}; no action executed")
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("Model response must be a JSON object")
            trace["response_id"] = body.get("id")
            trace["model"] = body.get("model", self.connection.model)
            trace["usage"] = body.get("usage", {})
            if "total_tokens" not in trace["usage"] and "input_tokens" in trace["usage"]:
                trace["usage"] = {**trace["usage"], "total_tokens": trace["usage"].get("input_tokens", 0)
                                  + trace["usage"].get("output_tokens", 0)}
            if body.get("choices"):
                trace["finish_reason"] = body["choices"][0].get("finish_reason")
                trace["output"] = body["choices"][0].get("message", {}).get("content")
            return body
        except httpx.TimeoutException as exc:
            trace["error_type"] = type(exc).__name__
            raise RuntimeError("Model request timed out; no action executed") from None
        except httpx.RequestError as exc:
            trace["error_type"] = type(exc).__name__
            raise RuntimeError("Model connection failed; no action executed") from None
        finally:
            trace["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)

    def choose(self, evidence, candidates, observations):
        self.trace = []
        started = time.perf_counter()
        menu = {c["id"]: c for c in candidates}
        probabilities = None
        intent = ""
        if self.provider == "astar":
            actions = [int(x) for x in self.astar.act(observations)]
            choice = ("a" if evidence.get("observation_mode") == "decentralized_local" else "j") + "".join(map(str, actions))
            intent = "Independent A* baseline; POGEMA resolves blocked moves"
        elif self.provider == "random":
            choice = candidates[int(self.rng.integers(len(candidates)))]["id"]
        elif self.provider == "qwen_rlcd":
            choice = self._qwen(evidence, candidates)
        elif self.provider == "jev":
            choice, probabilities = self._jev(evidence, candidates)
        else:
            payload = {"model": self.connection.model, "temperature": 0,
                               "max_tokens": self.connection.max_tokens, "response_format": {"type": "json_object"},
                               "messages": [{"role": "system", "content": instructions_for(evidence)},
                               {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}]}
            if self.provider == "deepseek":
                payload["thinking"] = {"type": "enabled" if self.connection.thinking else "disabled"}
                if self.connection.thinking:
                    payload["reasoning_effort"] = "low"
            body = self._post(payload)
            if body["choices"][0].get("finish_reason") == "length":
                raise ValueError("Output token budget exhausted; no action executed")
            if not body["choices"][0]["message"].get("content"):
                raise ValueError("Model returned no final answer; inspect finish_reason and token usage")
            answer = json.loads(body["choices"][0]["message"]["content"])
            if not isinstance(answer, dict):
                raise ValueError("Model answer must be a JSON object")
            choice = answer.get("choice")
            intent = str(answer.get("intent", ""))[:240]
            self.trace[-1]["answer"] = {"choice": choice, "intent": intent}
        if not isinstance(choice, str) or (self.provider != "astar" and choice not in menu):
            raise ValueError("Model selected an action outside the offered candidates; no action executed")
        if self.provider != "astar":
            actions = menu[choice]["actions"]
        return {"choice": choice, "actions": actions, "intent": intent,
                "provider": self.provider, "model_call": self.provider not in {"astar", "random"},
                "probabilities": probabilities, "admitted": choice in menu,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    def _qwen(self, evidence, candidates):
        """Condition each agent's <=5 labels on earlier choices; retain feasible completions."""
        remaining = candidates
        prefix = []
        for agent_id in range(len(candidates[0]["actions"])):
            available = sorted({c["actions"][agent_id] for c in remaining})
            labels = dict(zip("ABCDE", available))
            context = {"instructions": instructions_for(evidence), "state": evidence, "select_agent": agent_id,
                       "fixed_prefix": prefix, "labels": {k: self._qwen_label(evidence, v) for k, v in labels.items()}}
            body = self._post({"context": json.dumps(context), "schema": {"action": {
                "type": "enum", "choices": list(labels),
                "description": "Choose one label for this agent, respecting the fixed prefix"}},
                "temperature": 1.0})
            if body.get("is_valid_json") is not True or body.get("schema_match") is not True:
                raise ValueError("Qwen response failed schema validation")
            chosen = body["parsed_json"]["action"]["value"]
            scored = body["field_telemetry"]["action"]["candidate_probabilities"]
            if not isinstance(scored, list) or len(scored) != len(labels):
                raise ValueError("Qwen omitted candidate probabilities")
            probs = validate_probabilities({r["choice"]: r["probability"] for r in scored}, labels)
            if chosen not in labels:
                raise ValueError("Qwen selected an unoffered label")
            self.trace[-1]["answer"] = {"agent": agent_id, "label": chosen, "probabilities": probs}
            action = labels[chosen]
            prefix.append(action)
            remaining = [c for c in remaining if c["actions"][agent_id] == action]
        return remaining[0]["id"]

    @staticmethod
    def _qwen_label(evidence, action):
        # A small model reads label text far more reliably than cross-referencing the state JSON.
        facts = evidence.get("candidates", {}).get(f"a{action}")
        if evidence.get("observation_mode") != "decentralized_local" or not facts:
            return NAMES[action]
        return (f"{NAMES[action]}: destination {facts['destination_status']}, "
                f"own_map_distance_to_goal {facts.get('own_map_distance_to_goal')} "
                f"(change {facts.get('own_map_distance_change')})"
                + (", SUGGESTED by group plan" if facts.get("suggested_by_group_plan") else "")
                + (", WANTED by robot with right of way" if facts.get("wanted_by_peer_with_right_of_way") else "")
                + (", ENTERS cell of robot with right of way" if facts.get("enters_cell_of_peer_with_right_of_way") else "")
                + (", REVERSES your last step" if facts.get("reverses_last_successful_step") else "")
                + (f", same move already made {facts['times_same_transition_in_last_8_steps']}x recently"
                   if facts.get("times_same_transition_in_last_8_steps") else ""))

    def _jev_answer(self, state, criteria, instructions):
        if not 1 <= len(criteria) <= 255:
            raise ValueError("Jev Choice requires 1..255 options")
        body = self._post({"model": self.connection.model, "state": state,
                           "questions": {"action": {"type": "choice", "instructions": instructions,
                                                       "criteria": criteria}}})
        answer = body["answers"]["action"]
        probabilities = validate_probabilities(answer["probabilities"], criteria)
        choice = answer.get("choice")
        if choice not in criteria or probabilities[choice] + 1e-6 < max(probabilities.values()):
            raise ValueError("Jev choice must be an offered maximum-probability option")
        self.trace[-1]["answer"] = {"choice": choice, "probabilities": probabilities,
                                   "confidence": answer.get("confidence")}
        return choice, probabilities

    def _jev(self, evidence, candidates):
        instructions = instructions_for(evidence).split("Return only JSON:")[0] + " Select the best offered action for the next tick."
        if evidence.get("observation_mode") == "decentralized_local":
            return self._jev_answer(evidence, evidence["candidates"], instructions)
        if len(candidates) <= 255:
            criteria = {c["id"]: {"moves_by_agent_id": [NAMES[a] for a in c["actions"]],
                                   "next_positions_by_agent_id": c["destinations"]} for c in candidates}
            return self._jev_answer(evidence, criteria, instructions)
        # Do not truncate a joint menu to the provider's limit: every feasible action
        # remains reachable through a sequence of conditional <=5-way choices.
        remaining, prefix = candidates, []
        for agent_id in range(len(candidates[0]["actions"])):
            available = sorted({c["actions"][agent_id] for c in remaining})
            criteria = {NAMES[a]: {"agent_id": agent_id, "action": NAMES[a],
                          "next_position": next(c["destinations"][agent_id] for c in remaining
                                                if c["actions"][agent_id] == a)} for a in available}
            state = {**evidence, "select_agent": agent_id, "fixed_prefix_actions": list(prefix)}
            choice, _ = self._jev_answer(state, criteria, instructions +
                " This is a conditional choice for select_agent. Earlier agents' actions are fixed; "
                "each offered action still admits a collision-free completion for all remaining agents.")
            selected = NAMES.index(choice)
            prefix.append(selected)
            remaining = [c for c in remaining if c["actions"][agent_id] == selected]
        return remaining[0]["id"], None
