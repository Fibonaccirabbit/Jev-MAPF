"""Short-range messages between robots within sensor range (PIBT-style right of way).

A message carries only a priority and the sender's one-step wanted cells, as offsets from
the sender. Goals, memories and full plans are never shared. Absolute positions are used
only by the simulated radio to route messages and translate offsets into each receiver's
private frame, like a relative-bearing sensor would.
"""


def outgoing(evidence, not_at_goal_steps, tiebreak):
    xy = evidence["self_position"]
    wants = [[d[0] - xy[0], d[1] - xy[1]] for c in evidence["candidates"].values()
             if c.get("own_map_distance_change") == -1 for d in c["destinations"]]
    at_goal = evidence["at_own_goal"]
    # Higher tuple = right of way. Waiting longer since last at goal raises priority;
    # the fixed private tiebreak prevents two robots from yielding in lockstep.
    return {"priority": [int(not at_goal), 0 if at_goal else not_at_goal_steps, tiebreak],
            "at_goal": at_goal, "wants": wants}


def deliver(evidence, candidates, own_message, own_position, inbox, radius):
    """inbox: [(sender_absolute_position, message)] for all other robots; filtered by range here.

    Returns the enriched evidence/candidates and the highest priority asking for our cell, if any.
    """
    xy = evidence["self_position"]
    peers, requesters = [], []
    for position, message in inbox:
        rel = [position[0] - own_position[0], position[1] - own_position[1]]
        if max(abs(rel[0]), abs(rel[1])) > radius:
            continue
        where = [xy[0] + rel[0], xy[1] + rel[1]]
        peer = {"position": where, "at_goal": message["at_goal"],
                "has_right_of_way_over_you": message["priority"] > own_message["priority"],
                "wants_cells": [[where[0] + w[0], where[1] + w[1]] for w in message["wants"]]}
        if peer["has_right_of_way_over_you"] and list(xy) in peer["wants_cells"]:
            requesters.append(message["priority"])
        peers.append(peer)
    higher = [p for p in peers if p["has_right_of_way_over_you"]]
    wanted_by_higher = {tuple(c) for p in higher for c in p["wants_cells"]}
    wanted_by_lower = {tuple(c) for p in peers if not p["has_right_of_way_over_you"] for c in p["wants_cells"]}
    higher_cells = {tuple(p["position"]) for p in higher}
    for candidate in candidates:
        destination = tuple(candidate["destinations"][0])
        facts = {"wanted_by_peer_with_right_of_way": destination in wanted_by_higher,
                 "wanted_by_peer_you_outrank": destination in wanted_by_lower,
                 "enters_cell_of_peer_with_right_of_way": destination in higher_cells}
        candidate.update(facts)
        evidence["candidates"][candidate["id"]].update(facts)
    evidence["radio"] = {
        "your_priority": own_message["priority"],
        "priority_rule": "compare lists lexicographically: [not_at_goal, steps_since_last_at_goal, private_tiebreak]; higher has right of way",
        "neighbor_messages": peers,
        "yield_required": tuple(xy) in wanted_by_higher,
    }
    return evidence, candidates, max(requesters, default=None)


def inherit(evidence, candidates, message, requester_priority):
    """A robot asked to yield relays the requester's priority and advertises its escape cells."""
    xy = evidence["self_position"]
    escapes = [[d[0] - xy[0], d[1] - xy[1]] for c in candidates if c["actions"][0]
               and not c["wanted_by_peer_with_right_of_way"] and not c["enters_cell_of_peer_with_right_of_way"]
               for d in c["destinations"]]
    # Just below the requester, so the request still binds; above everyone the requester outranks.
    inherited = requester_priority[:-1] + [round(requester_priority[-1] - 1e-6, 7)]
    return {**message, "priority": max(message["priority"], inherited), "wants": escapes or message["wants"]}
