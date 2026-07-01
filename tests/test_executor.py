from retail_thor.executor import (
    build_open_receptacle_step,
    build_pick_place_steps,
    build_search_and_look_steps,
    build_visibility_recovery_actions,
    navigation_trace,
    nearest_reachable_position,
    verify_placement,
    yaw_toward,
)


def test_navigation_trace_selects_nearest_position_and_faces_target():
    positions = [{"x": 0, "y": 0, "z": 0}, {"x": 3, "y": 0, "z": 0}]
    target = {"position": {"x": 2, "y": 1, "z": 0}}

    trace = navigation_trace(positions, target)

    assert trace["chosen_reachable_position"] == {"x": 3, "y": 0, "z": 0}
    assert trace["distance_to_target"] == 1.0
    assert trace["sim_actions"][0]["action"] == "GetReachablePositions"
    assert trace["sim_actions"][1]["action"] == "Teleport"
    assert trace["sim_actions"][1]["rotation"]["y"] == yaw_toward({"x": 3, "y": 0, "z": 0}, target["position"])


def test_open_receptacle_step_only_opens_closed_openable_shelves():
    closed = {"source_object_id": "Fridge|1", "openable": True, "isOpen": False}
    already_open = {"source_object_id": "Fridge|1", "openable": True, "isOpen": True}

    assert build_open_receptacle_step(closed)["sim_actions"] == [{"action": "OpenObject", "objectId": "Fridge|1"}]
    assert build_open_receptacle_step(already_open) is None


def test_search_and_pick_place_steps_have_required_sim_actions():
    product = {"object_id": "Apple|1", "shelf_id": "shelf_001", "display_name_zh": "苹果"}

    search_steps = build_search_and_look_steps(product, "帮我找苹果")
    pick_place_steps = build_pick_place_steps(product, "Bowl|1", demo_mode=True)

    assert [step["action"] for step in search_steps] == ["search_object", "look_at"]
    assert search_steps[1]["sim_actions"][0]["action"] == "Teleport"
    assert pick_place_steps[0]["sim_actions"][0] == {
        "action": "PickupObject",
        "objectId": "Apple|1",
        "forceAction": True,
    }
    assert pick_place_steps[1]["sim_actions"][0] == {"action": "PutObject", "objectId": "Bowl|1"}


def test_visibility_recovery_and_placement_verification_are_bounded_and_metadata_based():
    recovery = build_visibility_recovery_actions(max_attempts=3)

    assert [action["action"] for action in recovery] == ["RotateRight", "RotateLeft", "LookDown"]
    assert verify_placement({"parentReceptacles": ["Bowl|1"]}, "Bowl|1") is True
    assert verify_placement({"parentReceptacles": ["CounterTop|1"]}, "Bowl|1") is False


def test_existing_geometry_helpers_still_work():
    assert nearest_reachable_position([{"x": 0, "z": 0}, {"x": 2, "z": 0}], {"x": 1.9, "z": 0}) == {
        "x": 2,
        "z": 0,
    }
