from jev_mapf import planner
from jev_mapf.policies import instructions_for
from jev_mapf.runtime import Episode


def test_groups_are_transitive_within_range():
    assert planner.groups([[0, 0], [0, 3], [0, 6], [9, 9]], 3) == [[0, 1, 2], [3]]


def test_joint_plan_uses_side_pocket_instead_of_swapping():
    # Corridor row 1 with a pocket at (0,1); robots must pass each other.
    cells = {(r, c): 1 for r in range(-1, 4) for c in range(-1, 5)}
    for c in range(4):
        cells[(1, c)] = 0
    cells[(0, 1)] = 0
    first, steps = planner.joint_first_step(cells, [(1, 0), (1, 3)], [(1, 3), (1, 0)])
    assert steps == 5 and len(first) == 2  # a swap-free pass must use the pocket


def test_no_suggestion_when_goal_unreachable():
    cells = {(0, 0): 0, (0, 1): 1, (-1, 0): 1, (1, 0): 1, (0, -1): 1}
    assert planner.joint_first_step(cells, [(0, 0), (5, 5)], [(0, 2), (5, 5)]) is None


def test_coop_off_by_default_and_suggestions_offered_when_on():
    off = Episode("puzzle-03", "random", 42, 8, num_agents=4)
    evidence, _ = off.policy.prepare(off.observations, 8, off.frames[-1]["positions"])
    assert off.config["coop_planner"] is False and "group_plan" not in evidence["agents"][0]
    assert "COOPERATIVE" not in instructions_for(evidence["agents"][0])
    off.close()
    on = Episode("puzzle-03", "random", 42, 8, num_agents=4, coop_planner=True)
    evidence, menus = on.policy.prepare(on.observations, 8, on.frames[-1]["positions"])
    for agent, menu in zip(evidence["agents"], menus):
        assert agent["group_plan"]["group_size"] == 4
        assert sum(c["suggested_by_group_plan"] for c in menu["options"]) == 1
        assert "COOPERATIVE PLANNING IS ENABLED" in instructions_for(agent)
    on.close()


def test_group_plan_supersedes_radio_yield():
    on = Episode("puzzle-00", "random", 42, 8, num_agents=4, coop_planner=True)
    evidence, _ = on.policy.prepare(on.observations, 8, on.frames[-1]["positions"])
    for agent in evidence["agents"]:
        if agent["group_plan"]["suggested_move"]:
            assert agent["radio"]["superseded_by_group_plan"] and not agent["radio"]["yield_required"]
    on.close()
