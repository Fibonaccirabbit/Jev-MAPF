import numpy as np

from jev_mapf.decentralized import DecentralizedPolicy, LocalMemory
from jev_mapf.perception import PROMPT_VERSION


def obs(xy=(0, 0), goal=(0, 2), radius=2):
    shape=(radius*2+1, radius*2+1)
    obstacles=np.zeros(shape, dtype=int)
    agents=np.zeros(shape, dtype=int)
    agents[radius, radius]=1
    return {"xy":np.array(xy),"target_xy":np.array(goal),"obstacles":obstacles,"agents":agents}


def test_overlay_coordinates_and_candidate_geometry():
    observation=obs(xy=(-3,4),goal=(-3,6))
    observation["obstacles"][1,2]=1
    observation["agents"][2,4]=1
    evidence, candidates=LocalMemory().prepare(observation,32)
    assert evidence["prompt_version"]==PROMPT_VERSION
    assert evidence["local_view"]["origin"]==[-5,2]
    assert evidence["local_view"]["rows"][2]=="..S.X"
    assert evidence["visible_peers"][0]["position"]==[-3,6]
    assert "local_obstacles" not in evidence and "local_agents_anonymous" not in evidence
    menu={c["id"]:c for c in candidates}
    assert "a1" not in menu
    assert menu["a4"]["destinations"]==[[-3,5]]
    assert menu["a4"]["goal_delta_after_move"]==[0,1]
    assert menu["a4"]["adjacent_cells_after_move"]["right"]=="another_agent"
    assert menu["a4"]["adjacent_cells_after_move"]["left"]=="vacated_self_cell"


def test_stationary_wait_and_blocked_are_distinct():
    p=DecentralizedPolicy("random",None,42,1)
    try:
        observation=obs()
        p.prepare([observation],32)
        p.feedback([observation],[observation],[0])
        p.feedback([observation],[observation],[4])
        evidence, _=p.prepare([observation],30)
        own=evidence["agents"][0]
        assert own["progress"]["consecutive_stationary_steps"]==2
        assert own["progress"]["consecutive_waits"]==0
        assert own["candidates"]["a4"]["recent_blocked_attempts_here"]==1
        assert own["candidates"]["a0"]["recent_blocked_attempts_here"]==0
        moved=obs(xy=(1,0))
        p.feedback([observation],[moved],[2])
        evidence,_=p.prepare([moved],29)
        assert evidence["agents"][0]["progress"]["consecutive_stationary_steps"]==0
    finally:
        p.close()


def test_trapped_neighbor_is_visible_fact_not_an_action_filter():
    observation=obs()
    observation["agents"][2,3]=1
    for r,c in ((1,3),(3,3),(2,4)):
        observation["obstacles"][r,c]=1
    evidence,candidates=LocalMemory().prepare(observation,32)
    assert evidence["visible_peers"][0]["currently_empty_exits"]==[]
    assert evidence["progress"]["adjacent_peer_without_empty_exit"]
    assert "a4" in {c["id"] for c in candidates}
    assert evidence["candidates"]["a4"]["destination_status"]=="another_agent"
    assert evidence["currently_empty_moves"]==["up","down","left"]


def test_outside_sensor_is_not_declared_empty_and_stale_agents_are_not_retained():
    memory=LocalMemory()
    observation=obs(radius=1)
    observation["agents"][1,2]=1
    memory.prepare(observation,32)
    observation=obs(xy=(0,-1),radius=1)
    evidence,_=memory.prepare(observation,31)
    assert evidence["visible_peers"]==[]
    assert evidence["candidates"]["a3"]["adjacent_cells_after_move"]["left"]=="not_currently_visible"
    assert evidence["candidates"]["a4"]["adjacent_cells_after_move"]["right"]=="not_currently_visible"


def test_candidate_menu_is_not_pruned_at_goal_or_during_stagnation():
    memory=LocalMemory()
    observation=obs(goal=(0,0))
    memory.history=[{"before":[0,0],"after":[0,0],"action":"wait","blocked":False} for _ in range(10)]
    evidence,candidates=memory.prepare(observation,22)
    assert evidence["at_own_goal"]
    assert evidence["progress"]["consecutive_stationary_steps"]==10
    assert len(candidates)==5
    assert [c["id"] for c in candidates]==["a0","a1","a2","a3","a4"]
    assert evidence["progress"]["at_goal_wait_is_success_not_stagnation"]
    assert not evidence["progress"]["two_cell_oscillation"]


def test_two_cell_loop_and_reversals_are_explicit_but_not_masked():
    memory=LocalMemory()
    memory.history=[
        {"before":[0,0],"after":[0,1],"action":"right","blocked":False},
        {"before":[0,1],"after":[0,0],"action":"left","blocked":False},
    ]*3
    evidence,candidates=memory.prepare(obs(goal=(3,3)),26)
    assert evidence["progress"]["two_cell_oscillation"]
    assert evidence["progress"]["unique_positions_in_last_8_steps"]==2
    assert evidence["candidates"]["a4"]["reverses_last_successful_step"]
    assert evidence["candidates"]["a4"]["times_same_transition_in_last_8_steps"]==3
    assert len(candidates)==5


def test_own_map_distance_routes_around_known_walls():
    # Wall directly between self and goal: straight-line delta says "right", BFS says detour.
    observation=obs(xy=(0,0),goal=(0,2))
    observation["obstacles"][0:4,3]=1
    evidence, candidates=LocalMemory().prepare(observation,32)
    menu={c["id"]:c for c in candidates}
    assert evidence["own_map_distance_to_goal"]==6
    assert "a4" not in menu and menu["a1"]["own_map_distance_change"]==1
    assert menu["a2"]["own_map_distance_change"]==-1
    assert menu["a0"]["own_map_distance_change"]==0
    assert evidence["candidates"]["a2"]["own_map_distance_to_goal"]==5


def test_own_map_distance_is_optimistic_about_unknown_and_ignores_robots():
    observation=obs(xy=(0,0),goal=(0,9))
    observation["agents"][2,3]=1
    evidence, candidates=LocalMemory().prepare(observation,32)
    menu={c["id"]:c for c in candidates}
    assert evidence["own_map_distance_to_goal"]==9
    assert menu["a4"]["own_map_distance_to_goal"]==8
    assert menu["a4"]["destination_status"]=="another_agent"


def test_own_map_distance_absent_when_goal_enclosed_by_known_walls():
    observation=obs(xy=(0,0),goal=(0,2),radius=3)
    for r,c in [(2,5),(4,5),(3,4),(3,6)]:
        observation["obstacles"][r,c]=1
    evidence, candidates=LocalMemory().prepare(observation,32)
    assert evidence["own_map_distance_to_goal"] is None
    assert all(c["own_map_distance_change"] is None for c in candidates)
