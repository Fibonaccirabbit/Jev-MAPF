"""Optional local cooperative planner for robots in radio contact (off by default).

Robots within radio range form a group and pool their own observed maps and goals. The
group runs a bounded joint A* (operator decomposition, POGEMA soft-collision rules: no
shared destination, no head-on swap, following allowed) and offers each member the first
step as a suggestion. The model still chooses; over budget there is no suggestion.
"""

import heapq

from .candidates import MOVES
from .perception import own_map_distances


def groups(positions, radius):
    parent = list(range(len(positions)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(positions):
        for j in range(i + 1, len(positions)):
            b = positions[j]
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= radius:
                parent[root(i)] = root(j)
    members = {}
    for i in range(len(positions)):
        members.setdefault(root(i), []).append(i)
    return sorted(members.values())


def joint_first_step(cells, starts, goals, budget=300000):
    """Return (first-step action per agent, plan length) or None when over budget/unsolvable."""
    rows = [p[0] for p in cells] + [p[0] for p in starts + goals]
    cols = [p[1] for p in cells] + [p[1] for p in starts + goals]
    r0, r1, c0, c1 = min(rows) - 1, max(rows) + 1, min(cols) - 1, max(cols) + 1
    dist = [own_map_distances(cells, g, s) for g, s in zip(goals, starts)]
    if any(tuple(s) not in d for s, d in zip(starts, dist)):
        return None
    n = len(starts)
    starts = tuple(tuple(s) for s in starts)
    goals = tuple(tuple(g) for g in goals)

    def free(p):
        return r0 <= p[0] <= r1 and c0 <= p[1] <= c1 and cells.get(p) != 1

    def h(base, partial):
        return (sum(dist[i][p] for i, p in enumerate(partial))
                + sum(dist[i][base[i]] for i in range(len(partial), n)))

    # Node: (f, tie, g, depth, base, partial, first_actions, partial_actions)
    tie = 0
    frontier = [(h(starts, ()), tie, 0, 0, starts, (), None, ())]
    best = {(starts, ()): 0}
    expansions = 0
    while frontier:
        f, _, g, depth, base, partial, first, acts = heapq.heappop(frontier)
        if not partial and base == goals:
            return (list(first) if first else [0] * n), depth
        expansions += 1
        if expansions > budget:
            return None
        i = len(partial)
        for action, (dr, dc) in enumerate(MOVES):
            new = (base[i][0] + dr, base[i][1] + dc)
            if not free(new) or new not in dist[i] or new in partial:
                continue
            if any(partial[j] == base[i] and new == base[j] for j in range(i)):
                continue
            step = 0 if base[i] == goals[i] and new == goals[i] else 1
            child_partial, child_acts = partial + (new,), acts + (action,)
            child_g, child_base, child_depth, child_first = g + step, base, depth, first
            if len(child_partial) == n:
                child_base, child_partial, child_depth = child_partial, (), depth + 1
                child_first, child_acts = first or child_acts, ()
            key = (child_base, child_partial)
            if best.get(key, float("inf")) <= child_g:
                continue
            best[key] = child_g
            tie += 1
            heapq.heappush(frontier, (child_g + h(child_base, child_partial), tie, child_g, child_depth,
                                      child_base, child_partial, child_first, child_acts))
    return None


def suggest(memories, evidences, positions, radio_range, budget=300000):
    """Per agent: (suggested action or None, group info)."""
    out = [None] * len(memories)
    for members in groups(positions, radio_range):
        # Each member shares its private map and goal, translated by relative position.
        cells, starts, goals = {}, [], []
        for i in members:
            xy = evidences[i]["self_position"]
            off = (positions[i][0] - xy[0], positions[i][1] - xy[1])
            for (r, c), wall in memories[i].cells.items():
                p = (r + off[0], c + off[1])
                cells[p] = max(cells.get(p, 0), wall)
            goal = evidences[i]["own_goal"]
            starts.append((positions[i][0], positions[i][1]))
            goals.append((goal[0] + off[0], goal[1] + off[1]))
        result = joint_first_step(cells, starts, goals, budget) if len(members) > 1 else None
        for k, i in enumerate(members):
            out[i] = {"group_size": len(members),
                      "action": result[0][k] if result else None,
                      "plan_steps": result[1] if result else None}
    return out
