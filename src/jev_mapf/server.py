"""Single-experiment browser workbench; controls remain responsive during model calls."""

import threading
import json
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .policies import Connection, PROVIDERS, default_connection, import_embodied_deepseek
from .runtime import Episode
from .tasks import CASES
from . import presets


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case: str = "crossing"
    provider: Literal["astar", "random", "deepseek", "chat", "qwen_rlcd", "jev"] = "astar"
    seed: int = Field(default=42, ge=0, lt=2**32)
    max_steps: int = Field(default=64, ge=1, le=256)
    mode: Literal["decentralized"] = "decentralized"
    obs_radius: int = Field(default=3, ge=1, le=5)
    num_agents: int | None = Field(default=None, ge=1, le=16)
    coop_planner: bool = False


def create_app(output_dir="outputs"):
    app = FastAPI(title="Jev-MAPF")
    web = Path(__file__).parent / "web"
    lock = threading.RLock()
    connections = {p: default_connection(p) for p in PROVIDERS if p not in {"astar", "random"}}
    verified = {p: False for p in connections}
    state = {"episode": None, "worker": None, "running": False}

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Pydantic's default errors can echo the whole connection object, including Key.
        return JSONResponse({"detail": [{"loc": list(e["loc"]), "type": e["type"]}
                                        for e in exc.errors()]}, status_code=422)

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin mutation rejected"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def bad_value(request, exc):
        return JSONResponse({"detail": "Invalid request or run state"}, status_code=400)

    @app.get("/")
    def index():
        return FileResponse(web / "index.html")

    @app.get("/api/config")
    def config():
        with lock:
            return {"cases": {k: {"name": v["name"], "description": v["description"],
                                  "official": "source" in v, "fixed_agents": len(v["agents_xy"]) if v.get("agents_xy") else None,
                                  "default_agents": v.get("num_agents", len(v.get("agents_xy", [])) or 4)} for k, v in CASES.items()},
                    "providers": list(PROVIDERS), "connections": {
                        p: {**c.public(), "verified": verified[p]} for p, c in connections.items()}}

    @app.put("/api/connections/{provider}")
    def configure(provider: str, value: Connection):
        with lock:
            if provider not in connections:
                raise HTTPException(404, "Unknown provider")
            if provider in {"deepseek", "jev"} and value.url != default_connection(provider).url:
                raise HTTPException(400, "Official provider requires its official endpoint")
            previous = connections[provider]
            if not value.key and value.url == previous.url:
                value = value.model_copy(update={"key": previous.key})
            connections[provider] = value
            verified[provider] = False
            return value.public()

    @app.post("/api/connections/import-deepseek")
    def import_saved():
        try:
            value = import_embodied_deepseek()
        except Exception:
            raise HTTPException(400, "无法读取原工作台的 DeepSeek 连接；可手动填写") from None
        with lock:
            connections["deepseek"] = value
            verified["deepseek"] = False
            return value.public()

    @app.post("/api/connections/{provider}/test")
    def probe(provider: str):
        with lock:
            if provider not in connections:
                raise HTTPException(404, "Unknown provider")
            connection = connections[provider].model_copy()
        trial = Episode(provider=provider, connection=connection, max_steps=1)
        try:
            result = trial.step()
            ok = result["status"] != "error" and result["metrics"]["model_calls"] > 0
            with lock:
                if connection == connections[provider]:
                    verified[provider] = ok
            return {"ok": ok, "model_calls": trial.policy.calls, "error": result["error"],
                    "decision": result["last_record"].get("decision"),
                    "calls": [{k: v for k, v in t.items() if k != "request"} for t in trial.policy.trace]}
        finally:
            trial.close()

    @app.get("/api/state")
    def snapshot():
        with lock:
            ep = state["episode"]
            return {"running": state["running"], "episode": ep.snapshot() if ep else None}

    def read_saved(name):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            raise HTTPException(400, "Invalid run name")
        directory = Path(output_dir) / name
        path = directory / "result.json"
        if directory.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            raise HTTPException(404, "Run not found")
        try:
            result = json.loads(path.read_text())
            if result.get("format") != "jev-mapf-run-v1":
                raise ValueError("Unsupported format")
            return result
        except (ValueError, OSError):
            raise HTTPException(400, "Unreadable experiment") from None

    @app.get("/api/runs")
    def saved_runs():
        result = []
        paths = sorted(Path(output_dir).glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths[:100]:
            try:
                run = read_saved(path.parent.name)
                result.append({"name": path.parent.name, "config": run["config"], "status": run["status"], "metrics": run["metrics"]})
            except (HTTPException, KeyError):
                continue
        return result

    @app.get("/api/presets")
    def list_presets():
        result = []
        for entry in presets.index():
            try:
                run = presets.load(entry["name"])
            except (KeyError, OSError, ValueError):
                continue
            result.append({**entry, "config": run["config"], "status": run["status"], "metrics": run["metrics"]})
        return result

    @app.get("/api/presets/{name}")
    def preset(name: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            raise HTTPException(400, "Invalid preset name")
        try:
            return presets.load(name)
        except (KeyError, OSError):
            raise HTTPException(404, "Preset not found") from None

    @app.get("/api/runs/{name}")
    def saved_run(name: str):
        return {**read_saved(name), "archive_name": name}

    @app.get("/api/runs/{name}/animation")
    def saved_animation(name: str):
        read_saved(name)
        path = Path(output_dir) / name / "replay.html"
        if path.is_symlink() or not path.is_file():
            raise HTTPException(404, "Animation not found")
        return FileResponse(path, media_type="text/html")

    def ensure_idle():
        worker = state["worker"]
        if worker and worker.is_alive():
            raise HTTPException(409, "A decision is in flight; pause and wait before resetting")

    @app.post("/api/reset")
    def reset(config: RunConfig):
        with lock:
            ensure_idle()
            ep = Episode(**config.model_dump(), connection=connections.get(config.provider))
            if state["episode"]:
                state["episode"].close()
            state.update(episode=ep, running=False)
            return ep.snapshot()

    def work(ep, continuous):
        try:
            while True:
                with lock:
                    if not state["running"]:
                        break
                    # Serialize the start with pause so a late request cannot clear a pause.
                    ep.cancel_pending.clear()
                ep.step(respect_pause=True)
                with lock:
                    if not continuous or not state["running"] or ep.status in {"success", "truncated", "error", "stopped"}:
                        break
        finally:
            with lock:
                state["running"] = False
            ep.save(Path(output_dir) / ep.id)

    @app.post("/api/control/{action}")
    def control(action: str):
        with lock:
            ep = state["episode"]
            if ep is None:
                raise HTTPException(400, "Create an experiment first")
            if action in {"pause", "stop"}:
                state["running"] = False
                ep.pause(stop=action == "stop")
            elif action in {"run", "step"}:
                ensure_idle()
                if ep.status in {"success", "truncated", "error", "stopped"}:
                    raise HTTPException(409, "Reset a terminal experiment before running")
                state["running"] = True
                worker = threading.Thread(target=work, args=(ep, action == "run"), daemon=True)
                state["worker"] = worker
                worker.start()
            else:
                raise HTTPException(404, "Unknown control")
            return {"accepted": True}

    @app.get("/api/export")
    def export():
        with lock:
            if state["episode"] is None:
                raise HTTPException(404, "No experiment")
            return state["episode"].snapshot(full=True)

    @app.get("/api/animation")
    def animation():
        with lock:
            ensure_idle()
            if state["episode"] is None:
                raise HTTPException(404, "No experiment")
            return HTMLResponse(str(state["episode"].env.render_html_animation()))

    app.mount("/static", StaticFiles(directory=web), name="static")
    return app
