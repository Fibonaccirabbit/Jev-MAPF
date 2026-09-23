"""Real POGEMA transitions, bounded execution and per-call experiment evidence."""

import copy
import json
import threading
import time
import uuid
from pathlib import Path

from pogema import pogema_v0

from .candidates import destinations, enumerate_candidates, legal_transition, model_input
from .policies import Policy
from .decentralized import DecentralizedPolicy
from .tasks import CASES, make_config
from .perception import PROMPT_VERSION


class Episode:
    def __init__(self, case="crossing", provider="astar", seed=42, max_steps=64, connection=None, policy=None,
                 mode="decentralized", obs_radius=3, num_agents=None, coop_planner=False):
        if mode not in {"decentralized", "centralized"}:
            raise ValueError("Invalid decision mode")
        self.decentralized = mode == "decentralized"
        self.id = uuid.uuid4().hex[:12]
        self.config = {"case": case, "provider": provider, "seed": seed, "max_steps": max_steps,
                       "observation_mode": "decentralized_local" if self.decentralized else "centralized_full_state",
                       "mode": mode, "obs_radius": obs_radius, "num_agents": num_agents, "on_target": "nothing",
                       "collision_system": "soft", "coop_planner": bool(coop_planner) and mode == "decentralized"}
        grid_config = make_config(case, seed, max_steps, obs_radius, num_agents)
        if not self.decentralized and grid_config.num_agents > 4:
            raise ValueError("Legacy centralized mode is limited to 4 agents")
        self.env = pogema_v0(grid_config)
        self.env.enable_animation()
        self.observations, _ = self.env.reset()
        self.config["num_agents"] = self.env.get_num_agents()
        if self.decentralized:
            self.config["prompt_version"] = PROMPT_VERSION
        if "source" in CASES[case]:
            self.config["map_source"] = copy.deepcopy(CASES[case]["source"])
        self.policy = policy or (DecentralizedPolicy(provider, connection, seed, self.env.get_num_agents(),
                                                     coop_planner=self.config["coop_planner"])
                                 if self.decentralized else Policy(provider, connection, seed))
        self.lock = threading.RLock()
        self.cancel_pending = threading.Event()
        self.busy = False
        self.status = "ready"
        self.error = None
        self.step_count = 0
        self.records = []
        self.frames = [self._state()]
        self.native_metrics = {}
        self.blocked_moves = 0
        self.created_at = time.time()

    def _state(self):
        grid = self.env.unwrapped.grid
        return {"map": grid.get_obstacles(ignore_borders=True).astype(int).tolist(),
                "positions": [[int(x) for x in p] for p in grid.get_agents_xy(ignore_borders=True)],
                "goals": [[int(x) for x in p] for p in grid.get_targets_xy(ignore_borders=True)],
                "step": self.step_count, "remaining_steps": self.config["max_steps"] - self.step_count}

    def _metrics(self):
        state = self.frames[-1]
        reached = [p == g for p, g in zip(state["positions"], state["goals"])]
        return {**self.native_metrics, "CSR": float(all(reached)), "ISR": sum(reached) / len(reached),
                "steps": self.step_count, "model_calls": self.policy.calls,
                "api_responses_200": sum(c.get("status_code") == 200 for r in self.records for c in r.get("calls", [])),
                "blocked_moves": self.blocked_moves,
                "api_latency_ms": round(sum(c.get("latency_ms", 0) for r in self.records
                                            for c in r.get("calls", [])), 2),
                "total_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for r in self.records
                                    for c in r.get("calls", [])),
                "decision_latency_ms": round(sum(r.get("decision", {}).get("latency_ms", 0)
                                                   for r in self.records), 2)}

    def snapshot(self, full=False):
        with self.lock:
            result = {"id": self.id, "config": self.config, "status": self.status,
                      "busy": self.busy, "error": self.error, "created_at": self.created_at,
                      "metrics": self._metrics(), "frames": self.frames,
                      "last_record": self.records[-1] if self.records else None}
            if full:
                result.update(format="jev-mapf-run-v1", records=self.records)
            return copy.deepcopy(result)

    def step(self, *, respect_pause=False):
        with self.lock:
            if respect_pause and self.cancel_pending.is_set():
                return self.snapshot()
            if self.busy or self.status in {"success", "truncated", "error", "stopped"}:
                raise ValueError("Episode is busy or terminal")
            if not respect_pause:
                self.cancel_pending.clear()
            self.busy = True
            self.status = "deciding"
            before = self.frames[-1]
            if self.decentralized:
                evidence, candidates = self.policy.prepare(self.observations, before["remaining_steps"],
                                                                before["positions"])
            else:
                candidates = enumerate_candidates(before["map"], before["positions"])
                evidence = model_input(before, candidates, [r for r in self.records if r.get("executed")])
            record = {"tick": self.step_count + 1, "before": before, "input": evidence,
                      "candidates": candidates, "executed": False}
        try:
            decision = self.policy.choose(evidence, candidates, self.observations)
            with self.lock:
                record["decision"] = decision
                record["calls"] = copy.deepcopy(self.policy.trace)
                if self.cancel_pending.is_set():
                    record["discarded"] = "Paused/stopped while model request was in flight"
                    self.records.append(record)
                    if self.status != "stopped":
                        self.status = "paused"
                    return self.snapshot()
                predicted = destinations(before["positions"], decision["actions"])
                # POGEMA's soft collision resolver mutates its action list in place.
                obs, rewards, terminated, truncated, infos = self.env.step(list(decision["actions"]))
                if self.decentralized:
                    self.policy.feedback(self.observations, obs, decision["actions"])
                self.observations = obs
                self.step_count += 1
                after = self._state()
                self.frames.append(after)
                self.blocked_moves += sum(a != b for a, b in zip(predicted, after["positions"]))
                record.update(after=after, executed=True, rewards=[float(r) for r in rewards],
                              terminated=[bool(v) for v in terminated], truncated=[bool(v) for v in truncated])
                self.records.append(record)
                if not legal_transition(before["map"], before["positions"], after["positions"]):
                    raise RuntimeError("POGEMA returned a collision; experiment aborted")
                if not self.decentralized and decision["model_call"] and predicted != after["positions"]:
                    raise RuntimeError("Candidate prediction disagreed with POGEMA; experiment aborted")
                self.native_metrics = infos[0].get("metrics", {})
                self.status = "success" if all(terminated) else "truncated" if all(truncated) else "paused"
        except Exception as exc:
            with self.lock:
                # Never include HTTP bodies, headers, connection reprs or secrets in error reports.
                self.error = f"{type(exc).__name__}: decision/execution failed; inspect call status and offered candidates"
                failed_calls = [c for c in self.policy.trace if "retry_after_seconds" not in c and
                                (c.get("error_type") or c.get("status_code", 200) != 200
                                 or c.get("finish_reason") == "length")]
                last_call = failed_calls[-1] if failed_calls else (self.policy.trace[-1] if self.policy.trace else {})
                if last_call.get("finish_reason") == "length":
                    self.error = "模型耗尽输出 token 预算，未执行动作；请增加预算或关闭思考模式后新建实验。"
                elif last_call.get("status_code", 200) != 200:
                    self.error = f"模型接口返回 HTTP {last_call['status_code']}，未执行动作；请检查连接配置。"
                elif last_call.get("error_type"):
                    self.error = f"模型请求失败（{last_call['error_type']}），未执行本轮动作；请检查网络连接。"
                record.update(error=self.error, calls=copy.deepcopy(self.policy.trace))
                if not self.records or self.records[-1] is not record:
                    self.records.append(record)
                self.status = "error"
        finally:
            with self.lock:
                self.busy = False
        return self.snapshot()

    def pause(self, stop=False):
        with self.lock:
            self.cancel_pending.set()
            if self.status not in {"success", "truncated", "error", "stopped"}:
                self.status = "stopped" if stop else "paused"

    def run(self):
        while self.status not in {"success", "truncated", "error", "stopped"}:
            self.step()
        return self.snapshot(full=True)

    def save(self, directory):
        directory = Path(directory)
        with self.lock:
            if self.busy:
                raise ValueError("Wait until the in-flight decision finishes before saving")
            directory.mkdir(parents=True, exist_ok=True)
            self.env.save_animation(str(directory / "animation.svg"))
            self.env.save_html_animation(str(directory / "replay.html"))
            (directory / "result.json").write_text(json.dumps(self.snapshot(full=True),
                                                            ensure_ascii=False, indent=2), encoding="utf-8")
        return directory

    def close(self):
        self.policy.close()
        self.env.close()
