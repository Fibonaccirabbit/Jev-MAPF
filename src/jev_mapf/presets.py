"""Bundled replay presets: slimmed copies of real runs (no request payloads, no secrets)."""

import json
from pathlib import Path

PRESET_DIR = Path(__file__).parent / "presets"
FACT_KEYS = ("move", "own_map_distance_change", "destination_status", "suggested_by_group_plan")
CALL_KEYS = ("agent_id", "latency_ms", "model", "status_code", "error_type", "retry_after_seconds", "answer")


def slim_run(run):
    records = []
    for r in run.get("records", []):
        agents = (r.get("input") or {}).get("agents") or []
        records.append({
            "tick": r["tick"], "executed": r.get("executed"), "error": r.get("error"),
            "decision": r.get("decision") and {
                "latency_ms": r["decision"].get("latency_ms"),
                "agents": [{k: d.get(k) for k in ("choice", "intent", "latency_ms", "probabilities")}
                           for d in r["decision"].get("agents", [])]},
            "candidates": [{"agent_id": c["agent_id"], "options": [{"id": o["id"], "actions": o["actions"]}
                                                                 for o in c["options"]]}
                           for c in r.get("candidates", [])],
            "input": {"agents": [{"at_own_goal": a.get("at_own_goal"), "group_plan": a.get("group_plan"),
                                  "candidates": {k: {f: v.get(f) for f in FACT_KEYS if f in v}
                                                 for k, v in a.get("candidates", {}).items()}}
                                 for a in agents]},
            "calls": [{**{k: c[k] for k in CALL_KEYS if k in c},
                       "usage": {"total_tokens": (c.get("usage") or {}).get("total_tokens", 0)}}
                      for c in r.get("calls", [])],
        })
    return {"format": run["format"], "id": run["id"], "config": run["config"], "status": run["status"],
            "error": run.get("error"), "metrics": run["metrics"], "frames": run["frames"],
            "last_record": records[-1] if records else None, "records": records}


def index():
    path = PRESET_DIR / "index.json"
    return json.loads(path.read_text()) if path.is_file() else []


def load(name):
    entry = next((p for p in index() if p["name"] == name), None)
    if entry is None:
        raise KeyError(name)
    return json.loads((PRESET_DIR / f"{name}.json").read_text())


def add(run_dir, name, title):
    run = json.loads((Path(run_dir) / "result.json").read_text())
    PRESET_DIR.mkdir(exist_ok=True)
    (PRESET_DIR / f"{name}.json").write_text(json.dumps(slim_run(run), ensure_ascii=False, separators=(",", ":")))
    entries = [p for p in index() if p["name"] != name] + [{"name": name, "title": title}]
    (PRESET_DIR / "index.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1))
