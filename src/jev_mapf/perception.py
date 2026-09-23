"""Readable facts derived only from one agent's sensors and executed history."""

from collections import deque

from .candidates import MOVES, NAMES

PROMPT_VERSION = "jev-mapf-local-v5"


def own_map_distances(cells, goal, xy):
    """BFS from own goal over own static memory; unknown cells are optimistically free.

    Robots are ignored, so this is a static-geometry fact, not a plan or reservation.
    """
    rows = [p[0] for p in cells] + [goal[0], xy[0]]
    cols = [p[1] for p in cells] + [goal[1], xy[1]]
    r0, r1, c0, c1 = min(rows) - 1, max(rows) + 1, min(cols) - 1, max(cols) + 1
    start = tuple(goal)
    distance = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVES[1:]:
            n = (r + dr, c + dc)
            if r0 <= n[0] <= r1 and c0 <= n[1] <= c1 and n not in distance and cells.get(n) != 1:
                distance[n] = distance[(r, c)] + 1
                queue.append(n)
    return distance


def enrich_local(evidence, candidates, cells, history):
    xy, goal = evidence["self_position"], evidence["own_goal"]
    radius = evidence["observation_radius"]
    obstacles = evidence.pop("local_obstacles")
    agents = evidence.pop("local_agents_anonymous")
    origin = [xy[0] - radius, xy[1] - radius]

    def at_sensor(p):
        r, c = p[0] - origin[0], p[1] - origin[1]
        return 0 <= r < len(obstacles) and 0 <= c < len(obstacles[0])

    def occupancy(p):
        if cells.get(tuple(p)) == 1:
            return "wall"
        if list(p) == xy:
            return "self"
        if not at_sensor(p):
            return "not_currently_visible"
        r, c = p[0] - origin[0], p[1] - origin[1]
        return "another_agent" if agents[r][c] else "empty"

    peers = []
    view = []
    for r, row in enumerate(obstacles):
        symbols = []
        for c, wall in enumerate(row):
            p = [origin[0] + r, origin[1] + c]
            other = bool(agents[r][c]) and p != xy
            symbols.append("#" if wall else "S" if p == xy else "X" if other and p == goal
                           else "A" if other else "G" if p == goal else ".")
            if other:
                empty_exits = []
                for move, (dr, dc) in enumerate(MOVES[1:], 1):
                    neighbor = [p[0] + dr, p[1] + dc]
                    if occupancy(neighbor) == "empty":
                        empty_exits.append({"move": NAMES[move], "position": neighbor})
                peers.append({"position": p, "currently_empty_exits": empty_exits,
                              "adjacent_to_self": abs(p[0]-xy[0])+abs(p[1]-xy[1]) == 1})
        view.append("".join(symbols))

    stationary = 0
    for result in reversed(history):
        if result["before"] != result["after"]:
            break
        stationary += 1
    waits = 0
    for result in reversed(history):
        if result["action"] != "wait":
            break
        waits += 1

    distance = own_map_distances(cells, goal, xy)
    current_distance = distance.get(tuple(xy))
    for candidate in candidates:
        destination = candidate["destinations"][0]
        after = distance.get(tuple(destination))
        candidate["own_map_distance_to_goal"] = after
        candidate["own_map_distance_change"] = (None if after is None or current_distance is None
                                                else after - current_distance)
        candidate["reverses_last_successful_step"] = bool(history and destination != xy
            and history[-1]["after"] != history[-1]["before"] and destination == history[-1]["before"])
        candidate["times_same_transition_in_last_8_steps"] = sum(
            h["before"] == xy and h["after"] == destination for h in history[-8:])
        candidate["goal_delta_after_move"] = [goal[0]-destination[0], goal[1]-destination[1]]
        candidate["destination_status"] = occupancy(destination)
        candidate["recent_blocked_attempts_here"] = sum(
            h["before"] == xy and h["action"] == candidate["move"] and h["blocked"]
            for h in history[-8:])
        # This is one-cell sensor annotation, not a route or action ranking.
        candidate["adjacent_cells_after_move"] = {
            NAMES[a]: ("vacated_self_cell" if [destination[0]+dr, destination[1]+dc] == xy
                        and destination != xy else occupancy([destination[0]+dr, destination[1]+dc]))
            for a, (dr, dc) in enumerate(MOVES[1:], 1)}

    evidence.update(
        prompt_version=PROMPT_VERSION,
        at_own_goal=xy == goal,
        local_view={"origin": origin, "rows": view,
                    "legend": "S=self; G=own goal; X=own goal occupied by another agent; A=anonymous agent; #=wall; .=empty"},
        visible_peers=peers,
        own_map_distance_to_goal=current_distance,
        progress={"consecutive_stationary_steps": stationary, "consecutive_waits": waits,
                  "at_goal_wait_is_success_not_stagnation": xy == goal,
                  "unique_positions_in_last_8_steps": len({tuple(h["after"]) for h in history[-8:]} | {tuple(xy)}),
                  "two_cell_oscillation": len(history) >= 4 and xy != goal and all(
                      h["before"] != h["after"] for h in history[-4:]) and len(
                      {tuple(h["before"]) for h in history[-4:]} | {tuple(h["after"]) for h in history[-4:]}) == 2,
                  "adjacent_peer_without_empty_exit": any(p["adjacent_to_self"] and not p["currently_empty_exits"] for p in peers)},
        currently_empty_moves=[c["move"] for c in candidates if c["actions"][0] and not c["visibly_occupied"]],
        candidates={c["id"]: {k: v for k, v in c.items() if k not in {"id", "actions"}} for c in candidates},
    )
    return evidence, candidates
