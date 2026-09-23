import argparse
import json
from pathlib import Path

from .policies import PROVIDERS, default_connection, import_embodied_deepseek
from .runtime import Episode
from .tasks import CASES


def main():
    parser = argparse.ArgumentParser(description="Jev-MAPF workbench and reproducible experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8091)
    run = sub.add_parser("run")
    run.add_argument("--case", choices=CASES, default="crossing")
    run.add_argument("--provider", choices=PROVIDERS, default="astar")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-steps", type=int, default=64)
    run.add_argument("--use-embodied-deepseek", action="store_true")
    run.add_argument("--output", type=Path)
    run.add_argument("--no-thinking", action="store_true", help="Disable DeepSeek thinking explicitly")
    run.add_argument("--max-tokens", type=int, default=8192)
    run.add_argument("--mode", choices=("decentralized", "centralized"), default="decentralized")
    run.add_argument("--obs-radius", type=int, default=3)
    run.add_argument("--num-agents", type=int)
    run.add_argument("--coop-planner", action="store_true",
                     help="Pool maps/goals within radio range and offer joint-plan suggestions")
    run.add_argument("--key-stdin", action="store_true", help="Read an API key from hidden terminal input")
    preset = sub.add_parser("preset", help="Bundle a saved run as a slim replay preset")
    preset.add_argument("run_dir", type=Path)
    preset.add_argument("--name", required=True)
    preset.add_argument("--title", required=True)
    args = parser.parse_args()
    if args.command == "preset":
        from .presets import add
        add(args.run_dir, args.name, args.title)
        return
    if args.command == "serve":
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
        return
    if args.use_embodied_deepseek and args.provider != "deepseek":
        parser.error("--use-embodied-deepseek requires --provider deepseek")
    connection = import_embodied_deepseek() if args.use_embodied_deepseek else default_connection(args.provider)
    connection = type(connection)(**{**connection.model_dump(), "thinking": not args.no_thinking,
                                    "max_tokens": args.max_tokens})
    if args.key_stdin:
        import getpass
        connection.key = getpass.getpass("API key (hidden): ")
    ep = Episode(args.case, args.provider, args.seed, args.max_steps, connection,
                 mode=args.mode, obs_radius=args.obs_radius, num_agents=args.num_agents,
                 coop_planner=args.coop_planner)
    try:
        while ep.status not in {"success", "truncated", "error", "stopped"}:
            result = ep.step()
            print(json.dumps({"step": result["metrics"]["steps"], "status": result["status"],
                              "choice": (result["last_record"].get("decision") or {}).get("choice"),
                              "ISR": result["metrics"]["ISR"],
                              "model_calls": result["metrics"]["model_calls"]}), flush=True)
        directory = ep.save(args.output or Path("outputs") / ep.id)
        print(json.dumps({"status": ep.status, "metrics": ep.snapshot()["metrics"], "output": str(directory)}))
        if ep.status != "success":
            raise SystemExit(1)
    finally:
        ep.close()


if __name__ == "__main__":
    main()
