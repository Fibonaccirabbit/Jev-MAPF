import numpy as np

from jev_mapf.decentralized import DecentralizedPolicy


def obs(radius=2, walls=(), agents=()):
    shape = (radius * 2 + 1, radius * 2 + 1)
    obstacles = np.zeros(shape, dtype=int)
    grid = np.zeros(shape, dtype=int)
    grid[radius, radius] = 1
    for r, c in walls:
        obstacles[r, c] = 1
    for r, c in agents:
        grid[r, c] = 1
    return obstacles, grid


def corridor(xy_rel_goal):
    # Horizontal corridor: only row `radius` is open.
    obstacles, grid = obs(walls=[(r, c) for r in (0, 1, 3, 4) for c in range(5)])
    return {"xy": np.array([0, 0]), "target_xy": np.array(xy_rel_goal), "obstacles": obstacles, "agents": grid}


def test_messages_translate_into_private_frames_and_bind_lower_priority():
    p = DecentralizedPolicy("random", None, 42, 2)
    try:
        a, b = corridor((0, 3)), corridor((0, -3))
        a["agents"][2, 3] = 1
        b["agents"][2, 1] = 1
        p.memories[0].steps_since_goal = 5  # waited longer => right of way
        evidence, menus = p.prepare([a, b], 32, positions=[[4, 10], [4, 11]])
        first, second = evidence["agents"]
        assert first["radio"]["neighbor_messages"][0]["position"] == [0, 1]
        assert second["radio"]["neighbor_messages"][0]["position"] == [0, -1]
        assert not first["radio"]["yield_required"] and second["radio"]["yield_required"]
        assert second["candidates"]["a0"]["wanted_by_peer_with_right_of_way"]
        assert second["candidates"]["a3"]["enters_cell_of_peer_with_right_of_way"]
        # The lower robot now advertises its escape cell, so the higher robot's way is clear.
        assert not first["candidates"]["a4"]["wanted_by_peer_with_right_of_way"]
        assert not first["candidates"]["a4"]["enters_cell_of_peer_with_right_of_way"]
        # The yielder relays inherited priority just below the requester and advertises its escape.
        relay = second["radio"]["your_broadcast"]
        assert first["radio"]["your_priority"] > relay["priority"] > [1, 0, 1.0]
        assert relay["wants"] == [[0, 1]]
    finally:
        p.close()


def test_radio_range_and_no_goal_leak():
    p = DecentralizedPolicy("random", None, 42, 2)
    try:
        a, b = corridor((0, 3)), corridor((0, -3))
        evidence, _ = p.prepare([a, b], 32, positions=[[0, 0], [0, 9]])
        assert evidence["agents"][0]["radio"]["neighbor_messages"] == []
        evidence, _ = p.prepare([a, b], 32, positions=[[0, 0], [0, 2]])
        message = evidence["agents"][0]["radio"]["neighbor_messages"][0]
        assert set(message) == {"position", "at_goal", "has_right_of_way_over_you", "wants_cells"}
        assert "own_goal" not in str(message) and [0, -1] not in message["wants_cells"]
    finally:
        p.close()


def test_without_positions_there_is_no_radio():
    p = DecentralizedPolicy("random", None, 42, 1)
    try:
        evidence, _ = p.prepare([corridor((0, 2))], 32)
        assert "radio" not in evidence["agents"][0]
    finally:
        p.close()
