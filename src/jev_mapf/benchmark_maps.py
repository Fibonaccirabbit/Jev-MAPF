"""Unmodified grid layouts from the official POGEMA benchmark (pinned commit)."""

import hashlib
import re
from pathlib import Path

SOURCE_COMMIT = "1209772eedc085fd2ce10dd459b7df60760aade7"
SOURCE_URL = "https://github.com/Cognitive-AI-Systems/pogema-benchmark"


def load_benchmark_maps():
    result = {}
    for category, dirname in (("mazes", "02-mazes"), ("warehouse", "03-warehouse"), ("puzzles", "05-puzzles")):
        path = Path(__file__).parent / "maps" / f"{category}.yaml"
        blocks, name = {}, None
        for line in path.read_text().splitlines():
            header = re.fullmatch(r'"([A-Za-z0-9_-]+)": \|-', line)
            if header:
                name = header.group(1)
                blocks[name] = []
            elif line.startswith("  ") and name:
                row = line[2:]
                if not re.fullmatch(r"[.#@$!]+", row):
                    raise ValueError(f"Unsupported benchmark row in {path.name}")
                blocks[name].append(row)
            elif line.strip():
                raise ValueError(f"Unsupported benchmark syntax in {path.name}")
        for name, rows in blocks.items():
            if not rows or len({len(r) for r in rows}) != 1:
                raise ValueError(f"Ragged benchmark map: {name}")
            grid = "\n".join(rows)
            result[name] = {"name": f"官方 {name}", "map": grid, "num_agents": 4,
                "description": f"POGEMA benchmark · {category} · {len(rows)}×{len(rows[0])} · 默认4智能体，可调整",
                "source": {"repository": SOURCE_URL, "commit": SOURCE_COMMIT,
                           "path": f"algorithms/experiments/{dirname}/maps.yaml", "map_name": name,
                           "grid_sha256": hashlib.sha256(grid.encode()).hexdigest()}}
    return result
