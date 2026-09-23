"""Benchmark definitions. All cases use classical, non-disappearing MAPF."""

from pogema import GridConfig
from .benchmark_maps import load_benchmark_maps

CASES = {
    "crossing": {
        "name": "双智能体交叉", "description": "6×6 开放地图，对角目标",
        "map": "......\n......\n......\n......\n......\n......",
        "agents_xy": [[0, 0], [0, 5]], "targets_xy": [[5, 5], [5, 0]],
    },
    "passing_bay": {
        "name": "狭道会车", "description": "单格走廊与避让口，需要让行再回到目标",
        "map": "#####\n##.##\n.....\n#####",
        "agents_xy": [[2, 0], [2, 4]], "targets_xy": [[2, 4], [2, 0]],
    },
    "four_rooms": {
        "name": "四智能体障碍图", "description": "8×8 地图，多方向通行",
        "map": "........\n..##....\n..##....\n........\n....##..\n....##..\n........\n........",
        "agents_xy": [[0, 0], [0, 7], [7, 0], [7, 7]],
        "targets_xy": [[7, 7], [7, 0], [0, 7], [0, 0]],
    },
    "random": {"name": "随机四智能体", "description": "8×8，障碍密度 0.2，固定 seed 可复现"},
}


CASES.update(load_benchmark_maps())


def make_config(case: str, seed: int = 42, max_steps: int = 64, obs_radius: int = 3, num_agents=None) -> GridConfig:
    if case not in CASES:
        raise ValueError("Unknown case")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(max_steps) is not int or not 1 <= max_steps <= 256:
        raise ValueError("max_steps must be an integer in [1, 256]")
    values = {k: v for k, v in CASES[case].items() if k not in {"name", "description", "source"}}
    if case == "random":
        values.update(size=8, num_agents=4, density=0.2)
    if type(obs_radius) is not int or not 1 <= obs_radius <= 5:
        raise ValueError("obs_radius must be an integer in [1, 5]")
    if num_agents is not None:
        if type(num_agents) is not int or not 1 <= num_agents <= 16:
            raise ValueError("num_agents must be in [1, 16]")
        if values.get("agents_xy") and len(values["agents_xy"]) != num_agents:
            raise ValueError("This fixed-start case has a fixed number of agents")
        values["num_agents"] = num_agents
    return GridConfig(**values, observation_type="POMAPF", on_target="nothing",
                      collision_system="soft", max_episode_steps=max_steps, seed=seed, obs_radius=obs_radius)
